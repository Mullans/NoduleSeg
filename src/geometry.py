import math
from typing import Optional, Sequence, Union

import gouda
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
from GoudaMI import ct_utils, vtk_utils, smart_image as si
from GoudaMI.convert import wrap_image_func

IMAGE_TYPE = Union[sitk.Image, si.SmartImage, np.ndarray]
RANDOM_TYPE = Optional[Union[int, npt.ArrayLike, np.random.Generator]]


# NOTE - Some of the functions in this file are adapted from https://github.com/norok2/raster_geometry
def get_bounds(mask: np.ndarray, bg_val: float = 0) -> list[tuple[int, int]]:
    """Get the corners of the bounding box/cube for the given binary label

    Returns
    -------
    List[Tuple[int, int]]
        A list of the [start, stop) indices for each axis - NOTE: inclusive start and exclusive stop
    """
    bounds = []
    if bg_val != 0:
        mask = mask != bg_val
    for i in range(mask.ndim):
        axis_check = np.any(mask, axis=tuple([j for j in range(mask.ndim) if j != i]))
        axis_range = np.where(axis_check == True) # noqa
        bounds.append([axis_range[0][0], axis_range[0][-1] + 1])
    return bounds


def find_point(shape: tuple, point: Union[float, Sequence] = 0.5, is_relative: bool = True) -> tuple:
    """Find the position of a point in the given shape

    Parameters
    ----------
    shape : tuple
        The shape to find the position in
    point : Union[float, Sequence], optional
        The point to find, by default 0.5
    is_relative : bool, optional
        Whether the point is relative to the shape or absolute, by default True
    """
    pos = gouda.force_len(point, len(shape))
    if is_relative:
        pos = tuple(gouda.rescale(x, 0, size - 1, 0, 1) for x, size in zip(pos, shape))
    return pos


def find_mesh(shape: tuple, point: Union[float, Sequence] = 0.5, is_relative: bool = True, dense: bool = False) -> Union[np.ndarray, list[np.ndarray]]:
    """Find the meshgrid of a point in the given shape

    Parameters
    ----------
    shape : tuple
        The shape to find the meshgrid in
    point : Union[float, Sequence], optional
        The point to center the meshgrid around, by default 0.5
    is_relative : bool, optional
        Whether the point is relative to the shape or absolute, by default True
    dense : bool, optional
        Whether to return a dense mesh-grid or an open mesh-grid, by default False
    """
    pos = find_point(shape, point, is_relative=is_relative)
    centered_slices = tuple(slice(-x, size - x) for x, size in zip(pos, shape))
    return np.mgrid[centered_slices] if dense else np.ogrid[centered_slices]


def x_rot2mat(x: float, as_degrees: bool = False) -> np.ndarray:
    """Return the 3D rotation matrix for a rotation around the x axis"""
    x = np.deg2rad(x) if as_degrees else x
    return np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])


def y_rot2mat(y: float, as_degrees: bool = False) -> np.ndarray:
    """Return the 3D rotation matrix for a rotation around the y axis"""
    y = np.deg2rad(y) if as_degrees else y
    return np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])


def z_rot2mat(z: float, as_degrees: bool = False) -> np.ndarray:
    """Return the 3D rotation matrix for a rotation around the z axis"""
    z = np.deg2rad(z) if as_degrees else z
    return np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])


def point_on_sphere(radius: float, theta: Optional[float] = None, phi: Optional[float] = None, random_seed: RANDOM_TYPE = None) -> np.ndarray:
    """Given the radius, theta, and phi polar coordinates, return the euclidean coordinates for a point on the sphere"""
    if isinstance(random_seed, np.random.Generator):
        random = random_seed
    else:
        random = np.random.default_rng(random_seed)
    theta = theta if theta is not None else random.random() * 2 * np.pi
    phi = phi if phi is not None else random.random() * np.pi
    x = radius * np.cos(theta) * np.sin(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(phi)
    return np.array([x, y, z])


def add_to_sphere_border(sphere_object: IMAGE_TYPE, item: IMAGE_TYPE, sphere_radius: float, theta: Optional[float] = None, phi: Optional[float] = None, combine: str = 'add', random_seed: RANDOM_TYPE = None) -> IMAGE_TYPE:
    """Add an item to the border of a sphere

    Parameters
    ----------
    sphere_object : IMAGE_TYPE
        The array/image of the sphere to add the item to
    item : IMAGE_TYPE
        The item to add the sphere
    sphere_radius : float
        The radius of the sphere
    theta : Optional[float], optional
        The theta spherical coordinate on the sphere surface (if None, selected randomly), by default None
    phi : Optional[float], optional
        The phi spherical coordinate on the sphere surface (if None, selected randomly), by default None
    combine : str, optional
        The method to use when combining objects with the sphere (either 'max' or 'add'), by default 'add'
    """
    assert isinstance(sphere_object, type(item)), f"output and item must have same type, found {type(sphere_object)} and {type(item)}"
    if isinstance(item, (sitk.Image, si.SmartImage)):
        item_shape = item.GetSize()
        output_shape = sphere_object.GetSize()
    else:
        item_shape = item.shape
        output_shape = sphere_object.shape

    surface_point = point_on_sphere(sphere_radius, theta=theta, phi=phi, random_seed=random_seed) + (np.array(output_shape) // 2)

    # Centered on surface_point
    merge_zone = tuple([slice(int(surface_point[i] - item_shape[i] // 2), int(surface_point[i] + (item_shape[i] - item_shape[i] // 2))) for i in range(len(surface_point))])

    if isinstance(item, (sitk.Image, si.SmartImage)):
        if isinstance(item, sitk.Image):
            temp = sphere_object[merge_zone]
            item.CopyInformation(temp)
        elif isinstance(item, si.SmartImage):
            temp = sphere_object.sitk_image[merge_zone]
            item.CopyInformation(temp)
            item = item.sitk_image
        if combine == 'max':
            sphere_object[merge_zone] = sitk.Maximum(temp, item)
        elif combine == 'add':
            sphere_object[merge_zone] = sitk.Add(temp, item)
        else:
            raise NotImplementedError(f'Combination mode `{combine}` not implemented.')
    elif isinstance(item, np.ndarray):
        if combine == 'max':
            sphere_object[merge_zone] = np.maximum(sphere_object[merge_zone], item)
        elif combine == 'add':
            sphere_object[merge_zone] += item
        else:
            raise NotImplementedError(f'Combination mode `{combine}` not implemented.')
    return sphere_object


def add_rotated_shape(sphere_img: si.SmartImage, horiz_cylinder: si.SmartImage, radius: float, theta: float, phi: float, random_seed: RANDOM_TYPE = None) -> si.SmartImage:
    """Rotate and add a shape to a sphere"""
    rotation = y_rot2mat(-phi) @ z_rot2mat(-theta)
    diag_cylinder = horiz_cylinder.euler_transform(rotation=rotation, in_place=False, interp=sitk.sitkLinear)
    sphere_img = add_to_sphere_border(sphere_img, diag_cylinder, radius, theta=theta, phi=phi, combine='max', random_seed=random_seed)
    return sphere_img


def make_superellipsoid(
        shape: tuple,
        radii: Union[float, Sequence] = 0.5,  # radius, so 0.5 is to the edge from center
        position: Union[float, Sequence] = 0.5,
        ndim: Optional[int] = None,
        use_rel_position: bool = True,
        use_rel_size: bool = True,
        smoothing: bool = False) -> np.ndarray:
    """Create an n-dimensional superellipsoid

    Parameters
    ----------
    shape : tuple
        The shape of the array to place the superellipsoid in
    radii : Union[float, Sequence], optional
        The radii of the superellipsoid to create along each dimension, by default 0.5
    ndim : Optional[int], optional
        The number of dimensions of the shape (if None, uses the max dimension among inputs), by default None
    use_rel_position : bool, optional
        Whether the position is relative to the array shape or absolute, by default True
    use_rel_size : bool, optional
        Whether the size is relative to the array shape or absolute, by default True
    smoothing : bool, optional
        Whether to smooth the border of the superellipsoid, by default False
    """
    if ndim is None:
        ndim = max([len(item) if gouda.is_iter(item) else 1 for item in (shape, position, radii)])
    (shape, position, semisize), _ = gouda.match_len(shape, position, radii, count=ndim)

    # get correct position
    radii = find_point(shape, radii, is_relative=use_rel_size)  # gets center coord of shape, only needs shape for ndim and if size is relative
    mesh = find_mesh(shape, position, is_relative=use_rel_position)

    rendered = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(mesh, radii):
        rendered += (np.abs(x_i / semisize) ** 2)
    if smoothing is False:
        rendered = rendered <= 1.0
    else:
        if smoothing > 0:
            k = math.prod(radii) ** (0.5 / ndim / smoothing)
            rendered = np.clip(1.0 - rendered, 0.0, 1.0 / k) * k
        else:
            rendered = rendered.astype(float)
    return rendered


def expand_shape(
        input_shape: np.ndarray,
        dim_size: int,
        dim_axis: int = -1,
        shape_size: float = 0.5,
        shape_position: float = 0.5,
        use_rel_position: bool = True,
        use_rel_size: bool = True) -> np.ndarray:
    """Expand a shape along a new axis

    Parameters
    ----------
    input_shape : np.ndarray
        The shape to expand
    dim_size : int
        Size of the new dimension
    dim_axis : int, optional
        Axis of the new dimension, by default -1
    shape_size : float, optional
        Size of the shape along the new dimension, by default 0.5
    shape_position : float, optional
        Position of the shape along the axis of the new dimension, by default 0.5
    use_rel_position : bool, optional
        Whether to use position as relative along the new axis or absolute, by default True
    use_rel_size : bool, optional
        Whether to use size as relative along the new axis or absolute, by default True
    """
    output_ndim = input_shape.ndim + 1
    if dim_axis > output_ndim:
        raise ValueError('dim_axis must be less than or equal to input_shape.ndim + 1')
    new_size = find_point((dim_size,), shape_size, is_relative=use_rel_size)[0]
    mesh = find_mesh((dim_size,), shape_position, is_relative=use_rel_position)[0]
    contains_shape = np.abs(mesh) <= (new_size / 2.0)
    output_shape = (input_shape.shape[:dim_axis] + (dim_size,) + input_shape.shape[dim_axis:])
    output_arr = np.zeros(output_shape, dtype=input_shape.dtype)
    for idx in range(dim_size):
        if contains_shape[idx]:
            new_dim_slice = [slice(None)] * output_ndim
            new_dim_slice[dim_axis] = idx
            output_arr[tuple(new_dim_slice)] = input_shape
    return output_arr


def make_cone(shape: tuple, base_radius: float = 0.5, height: float = 1.0, use_rel_sizes: bool = True) -> np.ndarray:
    """Make a 3D cone

    Parameters
    ----------
    shape : float
        Shape of the output array with the cone tapering along the last dimension
    base_radius : float
        Radius at the base of the cone
    height : float
        Height of the cone
    use_rel_sizes : bool
        Whether the base_radius and height are relative to the output array shape
    """
    shape = gouda.force_len(shape, 3)
    if use_rel_sizes:
        base_radius = min(shape[:2]) * base_radius
        height = shape[2] * height
    widths = base_radius * np.linspace(1, 0, num=height, endpoint=False)
    circles = [make_superellipsoid(shape, radii=widths[idx], position=0.5, ndim=2, use_rel_position=True, use_rel_size=False) for idx in range(height)]
    return np.stack(circles, axis=0)


def multiply_vector_image(image: sitk.Image, scalar: Union[float, Sequence[float]]) -> sitk.Image:
    """Multiply a vector image by a scalar

    Parameters
    ----------
    image : sitk.Image
        vector image to multiply
    scalar : Union[float, Sequence[float]]
        The scalar or list of scalars to multiply by
    """
    if not gouda.is_iter(scalar):
        scalar = (scalar,) * image.GetNumberOfComponentsPerPixel()
    assert len(scalar) == image.GetNumberOfComponentsPerPixel(), "Scalar must be a single value or a sequence of values the same length as the number of components in the image"
    return sitk.Compose([sitk.VectorIndexSelectionCast(image, i) * scalar[i % len(scalar)] for i in range(image.GetNumberOfComponentsPerPixel())])


def make_spiked_sphere(
        sphere_radius: Union[int, tuple[int, int]],
        spike_radius: Union[int, tuple[int, int]],
        spike_height: Union[int, tuple[int, int]],
        spike_count: int,
        isolate_spikes: bool = False,
        skip_spikes: bool = False,
        seed=None) -> tuple[si.SmartImage, si.SmartImage, int]:
    """Create a sphere with spikes on its surface

    Parameters
    ----------
    sphere_radius : Union[int, tuple[int, int]]
        The radius or (min, max) radii for the sphere
    spike_radius : Union[int, tuple[int, int]]
        The radius or (min, max) radii for the base of the spikes
    spike_height : Union[int, tuple[int, int]]
        The height or (min, max) for the height of the spikes
    spike_count : int
        The number of spikes to add to the sphere
    isolate_spikes : bool, optional
        If True, the first returned image is just the spikes, by default False
    skip_spikes : bool, optional
        If True, does not add any spikes to the sphere surface, by default False
    seed : _type_, optional
        The random seed or generator, by default None

    Returns
    -------
    si.SmartImage, si.SmartImage, int
        The spiked sphere image, the base sphere image, and the radius of the sphere
    """
    if isinstance(seed, np.random.Generator):
        random = seed
    else:
        random = np.random.default_rng(seed)
    if gouda.is_iter(sphere_radius):
        sphere_radius = random.integers(*sphere_radius)
    if not gouda.is_iter(spike_radius):
        spike_radius = (spike_radius, spike_radius + 1)
    if not gouda.is_iter(spike_height):
        spike_height = (spike_height, spike_height + 1)
    sphere = make_superellipsoid(sphere_radius * 2, radii=0.5, ndim=3)
    output = np.zeros([sphere_radius * 2 + spike_height[1] * 2] * 3, dtype=np.uint8)
    sphere_slice = tuple([slice(output.shape[i] // 2 - sphere.shape[i] // 2, output.shape[i] // 2 + sphere.shape[i] // 2) for i in range(3)])
    output[sphere_slice] = sphere
    sphere_img = si.SmartImage(output.copy())
    if isolate_spikes:
        spiked_img = si.SmartImage(np.zeros_like(output))
    else:
        spiked_img = si.SmartImage(output)

    if not skip_spikes:
        thetas = random.random(size=spike_count)
        thetas = np.cumsum(thetas / np.sum(thetas)) * 2 * np.pi
        random.shuffle(thetas)
        phis = random.random(size=spike_count)
        phis = np.cumsum(phis / np.sum(phis)) * np.pi
        random.shuffle(phis)
        radii = random.integers(*spike_radius, size=spike_count)
        heights = random.integers(*spike_height, size=spike_count)

        for theta, phi, radius, height in zip(thetas, phis, radii, heights):
            cone = make_cone(max(height, 2 * radius), radius, height, use_rel_sizes=False).astype(np.uint8)
            cone = si.SmartImage(cone)
            add_rotated_shape(spiked_img, cone, sphere_radius + 0.45 * height, theta=theta, phi=phi, random_seed=random)
    return spiked_img, sphere_img, sphere_radius


def make_spiked_nodule(
        core_sphere_radius: Union[int, tuple[int, int]] = (30, 50),
        core_spike_radius: Union[int, tuple[int, int]] = (2, 5),
        core_spike_height: Union[int, tuple[int, int]] = (20, 50),
        core_spike_count: int = 60,
        aux_sphere_radius: Union[int, tuple[int, int]] = (20, 30),
        aux_spike_radius: Union[int, tuple[int, int]] = (2, 5),
        aux_spike_height: Union[int, tuple[int, int]] = (20, 50),
        aux_spike_count: int = 20,
        num_spheres: int = 6,
        shift_type: str = 'avgRadius',
        deform: bool = True,
        deform_sigma: float = 2,
        deform_alpha: float = 2,
        smooth: bool = True,
        smooth_iter: int = 20,
        smooth_pass_band: float = 0.01,
        separate_spikes: bool = False,
        skip_spikes: bool = False,
        random_seed: RANDOM_TYPE = None) -> tuple[si.SmartImage, si.SmartImage]:
    """Create a nodule with spikes on its surface by combining multiple spiked spheres

    Parameters
    ----------
    core_sphere_radius : Union[int, tuple[int, int]], optional
        Radius or (min, max) radii for the central sphere of the nodule, by default (30, 50)
    core_spike_radius : Union[int, tuple[int, int]], optional
        Radius or (min, max) radii for the base of the spikes on the central sphere, by default (2, 5)
    core_spike_height : Union[int, tuple[int, int]], optional
        Height or (min, max) heights for the spikes on the central sphere, by default (20, 50)
    core_spike_count : int, optional
        Number of spikes on the central sphere, by default 60
    aux_sphere_radius : Union[int, tuple[int, int]], optional
        Radius or (min, max) radii for the non-central spheres of the nodule, by default (20, 30)
    aux_spike_radius : Union[int, tuple[int, int]], optional
        Radius or (min, max) radii for the base of the spikes on the non-central spheres, by default (2, 5)
    aux_spike_height : Union[int, tuple[int, int]], optional
        Height or (min, max) heights for the spikes on the non-central spheres, by default (20, 50)
    aux_spike_count : int, optional
        Number of spikes on the non-central spheres, by default 20
    num_spheres : int, optional
        Number of spheres to combine into the nodule, by default 6
    shift_type : str, optional
        Method to determine each sphere's shift from the center, by default 'avgRadius'
    deform : bool, optional
        Whether to apply elastic deformation to the final nodule surface (does not deform spikes), by default True
    deform_sigma : float, optional
        Sigma for the elastic deformation, by default 2
    deform_alpha : float, optional
        Alpha for the elastic deformation, by default 2
    smooth : bool, optional
        Whether to apply band-pass smoothing to the final nodule surface (does not smooth spikes), by default True
    smooth_iter : int, optional
        Number of iterations for smoothing, by default 20
    smooth_pass_band : float, optional
        Pass band for smoothing, by default 0.01
    separate_spikes : bool, optional
        If True, the first returned image is only spikes rather than a spiked nodule, by default False
    skip_spikes : bool, optional
        Whether to skip generating spikes for the nodule, by default False
    random_seed : RANDOM_TYPE, optional
        The random seed or generator, by default None

    Returns
    -------
    si.SmartImage, si.SmartImage
        The spiked sphere and base sphere images
    """

    if isinstance(random_seed, np.random.Generator):
        random = random_seed
    else:
        random = np.random.default_rng(random_seed)

    spiked_img, sphere_img, radius = make_spiked_sphere(core_sphere_radius, core_spike_radius, core_spike_height, core_spike_count, isolate_spikes=True, skip_spikes=skip_spikes, seed=random)
    spikes = []
    spheres = []
    radii = []
    upper_shifts = []
    lower_shifts = []

    for _ in range(num_spheres):
        spiked_img2, sphere_img2, radius2 = make_spiked_sphere(aux_sphere_radius, aux_spike_radius, aux_spike_height, aux_spike_count, isolate_spikes=True, skip_spikes=skip_spikes, seed=random)
        spikes.append(spiked_img2)
        spheres.append(sphere_img2)
        radii.append(radius2)
        if isinstance(shift_type, (int, float)):
            shift_size = shift_type
        elif shift_type == 'minRadius':
            shift_size = min(radius, radius2)
        elif shift_type == 'maxRadius':
            shift_size = max(radius, radius2)
        elif shift_type == 'avgRadius':
            shift_size = (radius + radius2) / 2
        shift = (random.random(size=3) * 2) - 1
        shift = np.ceil((shift / np.linalg.norm(shift)) * shift_size).astype(int)
        upper_shifts.append(np.maximum(shift, 0))
        lower_shifts.append(np.abs(np.minimum(shift, 0)))

    spikes = ct_utils.pad_to_same(spiked_img, *spikes, share_info=True)
    spheres = ct_utils.pad_to_same(sphere_img, *spheres, share_info=True)

    global_upper = np.max(upper_shifts, axis=0)
    global_lower = np.max(lower_shifts, axis=0)

    spiked_img = spikes[0].apply(sitk.ConstantPad, global_upper.tolist(), global_lower.tolist(), in_place=False)
    spiked_img.SetOrigin([0, 0, 0])
    spikes = spikes[1:]
    sphere_img = spheres[0].apply(sitk.ConstantPad, global_upper.tolist(), global_lower.tolist(), in_place=False)
    sphere_img.SetOrigin([0, 0, 0])
    spheres = spheres[1:]

    for idx in range(len(spikes)):
        spike = spikes[idx]
        sphere = spheres[idx]
        upper_shift = upper_shifts[idx]
        lower_shift = lower_shifts[idx]

        upper_pad = (global_upper - upper_shift) + lower_shift
        lower_pad = (global_lower - lower_shift) + upper_shift
        spike = spike.apply(sitk.ConstantPad, upper_pad.tolist(), lower_pad.tolist(), in_place=False)
        spike.SetOrigin([0, 0, 0])
        spiked_img = spiked_img | spike
        sphere = sphere.apply(sitk.ConstantPad, upper_pad.tolist(), lower_pad.tolist(), in_place=False)
        sphere.SetOrigin([0, 0, 0])
        sphere_img = sphere_img | sphere

    if smooth:
        sphere_img = vtk_utils.get_smoothed_contour(sphere_img, num_iterations=smooth_iter, pass_band=smooth_pass_band)
    if deform:
        sphere_img = ct_utils.elastic_deformation(sphere_img, sigma=deform_sigma, alpha=deform_alpha)
    spiked_img = spiked_img if separate_spikes else spiked_img | sphere_img
    spiked_img = ct_utils.pad_to_cube(spiked_img)
    spiked_img, sphere_img = ct_utils.pad_to_same(spiked_img, sphere_img, share_info=True)
    return spiked_img, sphere_img


def generate_sample(
        output_size: int = 250,
        spiked_nodule_args: Optional[dict] = None,
        randomize_intensity: bool = True,
        random_seed: RANDOM_TYPE = None):
    """Create a 2x2x2 grid of spiked nodules

    Parameters
    ----------
    output_size : int, optional
        Size along each dimension of the output image, by default 250
    spiked_nodule_args : Optional[dict], optional
        Optional keyword arguments for make_spiked_nodule, by default None
    randomize_intensity : bool, optional
        Whether to randomize the foreground intensity of each nodule, by default True
    random_seed : RANDOM_TYPE, optional
        The random seed or generator, by default None
    """
    img_max = wrap_image_func('sitk')(sitk.Maximum)
    if isinstance(random_seed, np.random.Generator):
        random = random_seed
    else:
        random = np.random.default_rng(random_seed)
    spiked_nodule_args = {} if spiked_nodule_args is None else spiked_nodule_args

    objects = []
    labels = []
    # Extra bit to ensure at least one spiked ball
    spike_zones = random.random(size=8) > 0.5
    while spike_zones.sum() == 0:
        spike_zones = random.random(size=8) > 0.5
    for i in range(8):
        has_spikes = spike_zones[i] > 0.5
        spike, sphere = make_spiked_nodule(random_seed=random, skip_spikes=not has_spikes, **spiked_nodule_args)

        # Sphere = 1, Spiked Sphere = 3, Spikes = 2
        if has_spikes:
            item = spike | sphere
            label = img_max(sphere * 3, spike * 2)
        else:
            item = sphere
            label = sphere
        label = si.SmartImage(label)
        item_bounds = ct_utils.get_bounds(item, bg_val=0)[1]
        bounds_slice = tuple([slice(b[0], b[1]) for b in item_bounds])
        item = item[bounds_slice].astype(np.float32)
        label = label[bounds_slice].astype(np.float32)

        if randomize_intensity:
            item = item * ((random.random() + 1) / 2)
        objects.append(item)
        labels.append(label)

    objects = ct_utils.pad_to_same(*objects, share_info=True)
    labels = ct_utils.pad_to_same(*labels, share_info=True)

    objects = [item.as_array() for item in objects]
    labels = [item.as_array() for item in labels]

    rows = [np.concatenate([objects[i], objects[i + 1]], axis=0) for i in range(0, 8, 2)]
    cols = [np.concatenate([rows[i], rows[i + 1]], axis=1) for i in range(0, 4, 2)]
    object_cube = np.concatenate(cols, axis=2)
    object_image = si.SmartImage(object_cube)
    object_image = ct_utils.pad_to_cube(object_image)

    rows = [np.concatenate([labels[i], labels[i + 1]], axis=0) for i in range(0, 8, 2)]
    cols = [np.concatenate([rows[i], rows[i + 1]], axis=1) for i in range(0, 4, 2)]
    label_cube = np.concatenate(cols, axis=2)
    label_image = si.SmartImage(label_cube)
    label_image = ct_utils.pad_to_cube(label_image)

    if output_size is not None:
        label_vec = ct_utils.label2vec(label_image)
        object_image = object_image.resample(size=(output_size,) * 3, outside_val=0, interp=sitk.sitkLinear)
        label_vec = label_vec.resample(size=(output_size,) * 3, outside_val=0, interp=sitk.sitkLinear)
        label_image = ct_utils.vec2label(label_vec)

    return object_image, label_image
