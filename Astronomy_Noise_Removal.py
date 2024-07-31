import numpy as np
import os
from astropy.io import fits
from astropy.nddata import CCDData
import ccdproc
import astroalign as aa
from astroscrappy import detect_cosmics
import re
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, AsinhStretch, LogStretch
import pywt
import cv2
from skimage.restoration import richardson_lucy, denoise_nl_means, estimate_sigma
from astropy.wcs import WCS
from reproject import reproject_interp

def load_all_fits_images(root_folder):
    files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith('.fit'):
                files.append(os.path.join(dirpath, f))
    files.sort(key=lambda x: int(re.findall(r'\d+', x.split('_')[-1].split('.')[0])[0]))
    return [(fits.open(file), file) for file in files]

def light_background_subtraction(image):
    sigma_clip = SigmaClip(sigma=2.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return image - bkg.background

def wavelet_denoise(image, wavelet='db1', level=3):
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(image.size))
    denoised_coeffs = [coeffs[0]] + [
        tuple(pywt.threshold(detail, uthresh, mode='soft') for detail in detail_tuple)
        for detail_tuple in coeffs[1:]
    ]
    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    return denoised_image

def apply_clahe(image, clip_limit=0.03, tile_grid_size=(32, 32)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image_clahe = clahe.apply(np.uint8(image * 255)) / 255.0
    return image_clahe

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    image_uint8 = np.uint8(image * 255)
    blurred = cv2.GaussianBlur(image_uint8, kernel_size, sigma)
    sharpened = float(amount + 1) * image_uint8 - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    low_contrast_mask = np.abs(image_uint8 - blurred) < threshold
    np.copyto(sharpened, image_uint8, where=low_contrast_mask)
    return sharpened / 255.0

def group_images_by_date(images):
    grouped_images = {}
    for hdulist, filename in images:
        date_obs = hdulist[0].header['DATE-OBS'].split('T')[0]
        if date_obs not in grouped_images:
            grouped_images[date_obs] = []
        grouped_images[date_obs].append((hdulist, filename))
    return grouped_images

def align_and_stack_images(images, reference_image=None, use_wcs=False):
    all_aligned_images = []
    wcs_ref = None if reference_image is None else WCS(reference_image.header)

    for science_ccd_hdulist, filename in images:
        try:
            science_ccd = CCDData(science_ccd_hdulist[0].data, unit='adu', meta=science_ccd_hdulist[0].header.copy())

            if reference_image is None:
                reference_image = science_ccd
                wcs_ref = WCS(reference_image.header)

            if use_wcs:
                aligned_data, _ = reproject_interp((science_ccd.data, WCS(science_ccd.header)), wcs_ref, shape_out=reference_image.shape)
            else:
                try:
                    aligned_data, _ = aa.register(science_ccd.data, reference_image.data, max_control_points=150, detection_sigma=2.5, min_area=4)
                except Exception as alignment_error:
                    print(f"Switching to WCS alignment due to error in astroalign for image {filename}: {alignment_error}")
                    use_wcs = True
                    aligned_data, _ = reproject_interp((science_ccd.data, WCS(science_ccd.header)), wcs_ref, shape_out=reference_image.shape)
                    for aligned_image in all_aligned_images:
                        data, header = aligned_image
                        aligned_data_wcs, _ = reproject_interp((data, WCS(header)), wcs_ref, shape_out=reference_image.shape)
                        aligned_image[0] = aligned_data_wcs

            all_aligned_images.append([aligned_data, science_ccd.header])

        except Exception as e:
            print(f"Skipping image {filename} due to error: {e}")

    if all_aligned_images:
        stack_data_with_bg_sub = np.sum([light_background_subtraction(image[0]) for image in all_aligned_images], axis=0)
        stack_data_with_bg_sub /= len(all_aligned_images)
        stacked_ccd_with_bg_sub = CCDData(stack_data_with_bg_sub.astype('float32'), unit='adu', meta=reference_image.meta)
        return stacked_ccd_with_bg_sub
    else:
        print("No images were successfully stacked.")
        return None

def calibrate_and_save_images(science_images, flat_images, bias_images, output_folder):
    bias_data = [CCDData(img[0].data, unit='adu') for img, _ in bias_images]
    combined_bias = ccdproc.combine(bias_data, method='median')

    flat_data = [CCDData(img[0].data, unit='adu') for img, _ in flat_images]
    corrected_flats = [ccdproc.subtract_bias(flat, combined_bias) for flat in flat_data]

    for flat in corrected_flats:
        cosmic_mask, clean_data = detect_cosmics(flat.data, gain=1.0, readnoise=10.0, sigclip=4.0)
        flat.data = clean_data

    normalized_flats = [flat.divide(np.median(flat.data[flat.data > 0])) for flat in corrected_flats]
    combined_flat = ccdproc.combine(normalized_flats, method='median')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    grouped_images = group_images_by_date(science_images)
    stacked_per_date = []

    for date, images in grouped_images.items():
        science_ccds = [CCDData(img[0].data, unit='adu', meta=img[0].header.copy()) for img, _ in images]
        calibrated_ccds = []

        for science_ccd in science_ccds:
            corrected_ccd = ccdproc.subtract_bias(science_ccd, combined_bias)
            corrected_ccd = ccdproc.flat_correct(corrected_ccd, combined_flat)
            cosmic_mask, clean_data = detect_cosmics(corrected_ccd.data, gain=1.0, readnoise=10.0, sigclip=4.0, sigfrac=0.3)
            corrected_ccd.data = clean_data
            calibrated_ccds.append(corrected_ccd)

        reference_image = None
        use_wcs = False

        stacked_ccd = align_and_stack_images([(ccd.to_hdu(), '') for ccd in calibrated_ccds], reference_image, use_wcs)
        if stacked_ccd:
            stacked_per_date.append(stacked_ccd)

    if stacked_per_date:
        reference_image = stacked_per_date[0]
        stacked_final = align_and_stack_images([(ccd.to_hdu(), '') for ccd in stacked_per_date], reference_image, True)

        if stacked_final:
            fits.writeto(os.path.join(output_folder, 'stacked_image_final.fits'), stacked_final.data, stacked_final.meta, overwrite=True)
            return stacked_final
    return None

def non_local_means_denoise(image):
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    patch_kw = dict(patch_size=5, patch_distance=6, channel_axis=None)
    denoised_image = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, **patch_kw)
    return denoised_image

def display_and_save_images(stacked_with_bg_sub, output_folder):
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))

    wavelet_denoised_data = wavelet_denoise(stacked_with_bg_sub.data)
    nlm_denoised_data = non_local_means_denoise(wavelet_denoised_data)
    nlm_denoised_data_only = non_local_means_denoise(stacked_with_bg_sub.data)

    norm_asinh_denoised = ImageNormalize(nlm_denoised_data, stretch=AsinhStretch(a=0.03))
    data_asinh_denoised = norm_asinh_denoised(nlm_denoised_data)

    norm_asinh_denoised2 = ImageNormalize(nlm_denoised_data_only, stretch=AsinhStretch(a=0.03))
    data_asinh_denoised2 = norm_asinh_denoised2(nlm_denoised_data_only)

    norm_log_denoised = ImageNormalize(nlm_denoised_data, stretch=LogStretch(a=30))
    data_log_denoised = norm_log_denoised(nlm_denoised_data)

    psf1 = np.ones((5, 5)) / 25
    psf2 = np.ones((7, 7)) / 49
    deblurred_asinh_denoised = richardson_lucy(data_asinh_denoised, psf1, num_iter=10)
    deblurred_log_denoised = richardson_lucy(data_log_denoised, psf2, num_iter=10)

    asinh_denoised = richardson_lucy(data_asinh_denoised2, psf1, num_iter=15)
    log_denoised = richardson_lucy(wavelet_denoised_data, psf2, num_iter=10)

    images = [
        (data_asinh_denoised, 'wavelet_and_nonlocalmean_Denoised + Asinh Stretched'),
        (data_log_denoised, 'wavelet_and_nonlocalmean_Denoised + Log Stretched'),
        (deblurred_asinh_denoised, 'wavelet_and_nonlocalmean_Denoised + Asinh Stretched +  richardson_lucy_deblurred'),
        (deblurred_log_denoised, 'wavelet_Denoised + Log Stretched +  richardson_lucy_deblurred'),
        (asinh_denoised, 'non-local means denoising+ Asinh Stretched +  richardson_lucy_deblurred'),
    ]

    for idx, (image, title) in enumerate(images):
        row, col = divmod(idx, 2)
        axes[row, col].imshow(image, origin='lower', cmap='gray')
        axes[row, col].set_title(title)
        fits.writeto(os.path.join(output_folder, f'stacked_image_{title.lower().replace(" ", "_")}.fits'), image, stacked_with_bg_sub.meta, overwrite=True)

    plt.tight_layout()
    plt.show()

science_folder = "D:\\Astronomy-Image-Processing\\m51\\m51_b"
flat_folder = "D:\\Astronomy-Image-Processing\\m51\\flat_b"
bias_folder = "D:\\Astronomy-Image-Processing\\m51\\bias"
output_folder = "D:\\Astronomy-Image-Processing\\m51\\calibrated_images_m51_b"

flat_images = load_all_fits_images(flat_folder)
bias_images = load_all_fits_images(bias_folder)
science_images = load_all_fits_images(science_folder)

stacked_ccd_with_bg_sub = calibrate_and_save_images(science_images, flat_images, bias_images, output_folder)

if stacked_ccd_with_bg_sub:
    display_and_save_images(stacked_ccd_with_bg_sub, output_folder)

print("Calibration and stacking complete.")
