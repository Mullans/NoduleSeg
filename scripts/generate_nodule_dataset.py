import gouda
import nnunet.paths
import numpy as np
import SimpleITK as sitk
import tqdm.auto as tqdm
from GoudaMI import smart_image as si, ct_utils
from nnunet.dataset_conversion.utils import generate_dataset_json

from src.geometry import generate_sample, RANDOM_TYPE
from src.noise import generate_fractal_noise_3d


def prep_single_sample(base_dir: gouda.GoudaPath, data_type: str, sample_idx: int, shape: int = 128, seed: RANDOM_TYPE = None, randomize_intensity: bool = True, noise_scaling: float = 0.5) -> None:
    image_dir = base_dir(f'images{data_type}').ensure_dir()
    label_dir = base_dir(f'labels{data_type}').ensure_dir()
    full_label_dir = base_dir(f'labels{data_type}_full').ensure_dir()

    if isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)

    nodule, label = generate_sample(output_size=shape, randomize_intensity=randomize_intensity, random_seed=random)
    nodule_arr = nodule.as_array()
    label_arr = label.as_array()
    bounds = gouda.image.get_bounds(nodule_arr)
    shift_range = [[-bounds[idx][0], shape - bounds[idx][1]] for idx in range(nodule_arr.ndim)]
    shift = [random.integers(*r) if r[1] - r[0] > 0 else 0 for r in shift_range]
    nodule_arr = np.roll(nodule_arr, shift, axis=(0, 1, 2))
    label_arr = np.roll(label_arr, shift, axis=(0, 1, 2))
    nodule = si.SmartImage(nodule_arr)
    label = si.SmartImage(label_arr)

    rotation = random.random(size=3) * np.pi * 2
    label_vec = ct_utils.label2vec(label, bg_val=0)
    nodule = nodule.euler_transform(rotation=rotation, interp=sitk.sitkLinear)
    label_vec = label_vec.euler_transform(rotation=rotation, interp=sitk.sitkLinear)
    label = ct_utils.vec2label(label_vec)

    full_label_image = label.astype(np.uint8)

    full_label_image.astype(np.uint8).write(full_label_dir('nodule_{:03d}.nii.gz'.format(sample_idx)), image_type='sitk')

    (label == 3).astype(np.uint8).write(label_dir('nodule_{:03d}.nii.gz'.format(sample_idx)), image_type='sitk')

    noise = generate_fractal_noise_3d(nodule.GetSize()[::-1], 5, octaves=3, seed=random).astype(np.float32)
    noise = noise * noise_scaling
    noise = si.SmartImage(noise)
    merged: si.SmartImage = noise + nodule
    merged.window(-1, 1, in_place=True)
    merged.write(image_dir('nodule_{:03d}_0000.nii.gz'.format(sample_idx)), image_type='sitk')


if __name__ == '__main__':
    num_samples = 100
    task_name = "Task902_NoduleSeg"

    base_dir = gouda.GoudaPath(nnunet.paths.nnUNet_raw_data) / task_name

    random = np.random.default_rng(42)
    for sample_idx in tqdm.trange(num_samples, desc='Generating training samples'):
        prep_single_sample(base_dir, 'Tr', sample_idx, seed=random)

    random = np.random.default_rng(84)
    for sample_idx in tqdm.trange(num_samples, num_samples + num_samples, desc='Generating test samples'):
        prep_single_sample(base_dir, 'Ts', sample_idx, seed=random)

    labels = {
        "0": "background",
        "1": "SpikedLump",
    }
    json_path = base_dir('dataset.json')
    generate_dataset_json(json_path, base_dir('imagesTr').path, base_dir('imagesTs').path, ['None', ], labels, task_name.split('_')[1])
