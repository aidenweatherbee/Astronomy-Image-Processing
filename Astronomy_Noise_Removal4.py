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
import pywt  # Import PyWavelets for wavelet denoising
import cv2  # Import OpenCV for CLAHE
from skimage.restoration import richardson_lucy  # Import Richardson-Lucy deconvolution from scikit-image

def load_all_fits_images(root_folder):
    """Recursively load all FITS files from a specified root folder and return both the HDUList and filenames."""
    files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for f in filenames:
            if f.endswith('.fit'):
                files.append(os.path.join(dirpath, f))
    files.sort(key=lambda x: int(re.findall(r'\d+', x.split('_')[-1].split('.')[0])[0]))  # Sort numerically
    return [(fits.open(file), file) for file in files]

def light_background_subtraction(image):
    """Perform light background subtraction suitable for galaxy images."""
    sigma_clip = SigmaClip(sigma=2.0)  # Adjust sigma clipping for better background subtraction
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return image - bkg.background

def wavelet_denoise(image, wavelet='db1', level=3):  # Increase the wavelet decomposition level
    """Perform wavelet denoising on the image."""
    coeffs = pywt.wavedec2(image, wavelet, level=level)
    sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    uthresh = sigma * np.sqrt(2 * np.log(image.size))
    
    # Apply soft thresholding and convert generator objects to lists or tuples
    denoised_coeffs = [coeffs[0]] + [
        tuple(pywt.threshold(detail, uthresh, mode='soft') for detail in detail_tuple) 
        for detail_tuple in coeffs[1:]
    ]

    denoised_image = pywt.waverec2(denoised_coeffs, wavelet)
    return denoised_image

def apply_clahe(image, clip_limit=0.03, tile_grid_size=(32, 32)):
    """Apply CLAHE to an image with reduced effect."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    image_clahe = clahe.apply(np.uint8(image * 255)) / 255.0  # Convert to uint8, apply CLAHE, and convert back
    return image_clahe

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.5, threshold=0):
    """Apply Unsharp Masking to enhance details in the image."""
    # Ensure the image is in uint8 format
    image_uint8 = np.uint8(image * 255)
    blurred = cv2.GaussianBlur(image_uint8, kernel_size, sigma)
    sharpened = float(amount + 1) * image_uint8 - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    low_contrast_mask = np.abs(image_uint8 - blurred) < threshold
    np.copyto(sharpened, image_uint8, where=low_contrast_mask)
    return sharpened / 255.0

def calibrate_and_save_images(science_images, flat_images, bias_images, output_folder):
    """Calibrate science images using provided flat and bias frames, perform cosmic ray removal, realign images, stack all images together, and save the calibrated images."""
    print(f"Number of bias images: {len(bias_images)}")
    print(f"Number of flat images: {len(flat_images)}")
    
    bias_data = [CCDData(img[0].data, unit='adu') for img, _ in bias_images]
    combined_bias = ccdproc.combine(bias_data, method='median')

    flat_data = [CCDData(img[0].data, unit='adu') for img, _ in flat_images]
    corrected_flats = [ccdproc.subtract_bias(flat, combined_bias) for flat in flat_data]
    
    for flat in corrected_flats:
        cosmic_mask, clean_data = detect_cosmics(flat.data, gain=1.0, readnoise=10.0, sigclip=4.0)  # Adjust sigclip for better cosmic ray detection
        flat.data = clean_data

    normalized_flats = [flat.divide(np.median(flat.data[flat.data > 0])) for flat in corrected_flats]
    combined_flat = ccdproc.combine(normalized_flats, method='median')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    science_ccds = [CCDData(img[0].data, unit='adu', meta=img[0].header.copy()) for img, _ in science_images]

    reference_image = None
    exposure_time = science_images[0][0][0].header['EXPTIME']
    print(f"Exposure time of each image: {exposure_time} seconds")

    stack_data_with_bg_sub = np.zeros_like(science_ccds[0].data, dtype='float32')
    stack_header = science_ccds[0].meta

    successful_stacks = 0
    failed_stacks = 0

    for (science_ccd_hdulist, filename) in science_images:
        try:
            # Ensure we are working with CCDData
            science_ccd = CCDData(science_ccd_hdulist[0].data, unit='adu', meta=science_ccd_hdulist[0].header.copy())

            corrected_ccd = ccdproc.subtract_bias(science_ccd, combined_bias)
            corrected_ccd = ccdproc.flat_correct(corrected_ccd, combined_flat)

            cosmic_mask, clean_data = detect_cosmics(corrected_ccd.data, gain=1.0, readnoise=10.0, sigclip=4.0, sigfrac=0.3)  # Adjust sigclip for better cosmic ray detection
            corrected_ccd.data = clean_data

            if reference_image is None:
                reference_image = corrected_ccd

            try:
                aligned_data, _ = aa.register(corrected_ccd.data, reference_image.data, max_control_points=150, detection_sigma=2.5, min_area=4)
            except Exception as alignment_error:
                print(f"Retrying alignment for image {filename} with adjusted parameters due to error: {alignment_error}")
                aligned_data, _ = aa.register(corrected_ccd.data, reference_image.data, max_control_points=100, detection_sigma=1.5, min_area=3)

            aligned_ccd = CCDData(aligned_data, unit='adu', meta=corrected_ccd.meta)

            # Apply light background subtraction after alignment
            background_subtracted_data = light_background_subtraction(aligned_ccd.data)

            stack_data_with_bg_sub += background_subtracted_data
            successful_stacks += 1
        except Exception as e:
            print(f"Skipping image {filename} due to error: {e}")
            failed_stacks += 1

    if successful_stacks > 0:
        stack_data_with_bg_sub /= successful_stacks

        stacked_ccd_with_bg_sub = CCDData(stack_data_with_bg_sub.astype('float32'), unit='adu', meta=stack_header)

        # Save the stacked images
        fits.writeto(os.path.join(output_folder, 'stacked_image_with_bg_sub.fits'), stacked_ccd_with_bg_sub.data, stacked_ccd_with_bg_sub.meta, overwrite=True)

        print(f"Number of successfully stacked images: {successful_stacks}")
    else:
        print("No images were successfully stacked.")

    if successful_stacks > 0:
        return stacked_ccd_with_bg_sub
    else:
        return None

def display_and_save_images(stacked_with_bg_sub, output_folder):
    """Display and save the stacked images after auto-stretching using advanced stretching."""
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))

    # Apply wavelet denoising
    denoised_data = wavelet_denoise(stacked_with_bg_sub.data)

    # Asinh Stretched Image on Denoised Data
    norm_asinh_denoised = ImageNormalize(denoised_data, stretch=AsinhStretch(a=0.8))  # Adjust the intensity of AsinhStretch
    data_asinh_denoised = norm_asinh_denoised(denoised_data)

    # Apply CLAHE to Asinh Stretched Data
    data_asinh_denoised_clahe = apply_clahe(data_asinh_denoised, clip_limit=0.9, tile_grid_size=(12, 12))

    # Log Stretched Image on Denoised Data
    norm_log_denoised = ImageNormalize(denoised_data, stretch=LogStretch(a=55))  # Adjust the intensity of LogStretch
    data_log_denoised = norm_log_denoised(denoised_data)

    # Apply CLAHE to Log Stretched Data with more conservative parameters
    data_log_denoised_clahe = apply_clahe(data_log_denoised, clip_limit=0.3, tile_grid_size=(64, 64))

    # Apply Unsharp Masking to CLAHE applied images with more conservative parameters
    data_asinh_denoised_clahe_sharpened = unsharp_mask(data_asinh_denoised_clahe, amount=1.5)
    data_log_denoised_clahe_sharpened = unsharp_mask(data_log_denoised_clahe, amount=1.5)

    # Deblurring using Richardson-Lucy deconvolution
    psf = np.ones((5, 5)) / 25  # Example PSF, adjust as needed
    deblurred_asinh_denoised_clahe = richardson_lucy(data_asinh_denoised_clahe, psf, num_iter=30)
    deblurred_log_denoised_clahe = richardson_lucy(data_log_denoised_clahe, psf, num_iter=10)

    # Display and save images
    images = [
        (data_asinh_denoised_clahe, 'Denoised + Asinh Stretched + CLAHE'),
        (data_asinh_denoised_clahe_sharpened, 'Denoised + Asinh Stretched + CLAHE + Sharpened'),
        (data_log_denoised_clahe, 'Denoised + Log Stretched + CLAHE'),
        (data_log_denoised_clahe_sharpened, 'Denoised + Log Stretched + CLAHE + Sharpened'),
        (deblurred_asinh_denoised_clahe, 'Deblurred + Asinh Stretched + CLAHE'),
        (deblurred_log_denoised_clahe, 'Deblurred + Log Stretched + CLAHE')
    ]

    for idx, (image, title) in enumerate(images):
        row, col = divmod(idx, 2)
        axes[row, col].imshow(image, origin='lower', cmap='gray')
        axes[row, col].set_title(title)
        fits.writeto(os.path.join(output_folder, f'stacked_image_{title.lower().replace(" ", "_")}.fits'), image, stacked_with_bg_sub.meta, overwrite=True)

    plt.tight_layout()
    plt.show()

# Example usage
science_folder = "D:\\Astronomy-Image-Processing\\m101\\m101_r"
flat_folder = "D:\\Astronomy-Image-Processing\\m101\\flat_r"
bias_folder = "D:\\Astronomy-Image-Processing\\m101\\bias"
output_folder = "D:\\Astronomy-Image-Processing\\m101\\calibrated_images_m101_r"

flat_images = load_all_fits_images(flat_folder)
bias_images = load_all_fits_images(bias_folder)
science_images = load_all_fits_images(science_folder)

stacked_ccd_with_bg_sub = calibrate_and_save_images(science_images, flat_images, bias_images, output_folder)

if stacked_ccd_with_bg_sub:
    display_and_save_images(stacked_ccd_with_bg_sub, output_folder)

print("Calibration and stacking complete.")
