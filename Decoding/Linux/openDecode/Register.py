from skimage.registration import phase_cross_correlation
import numpy as np
import glob
import cv2
import os

import SimpleITK as sitk

def _getRegisterMatrix(ref:np.ndarray, dst: np.ndarray):
    shift, _, _ = phase_cross_correlation(ref, dst, upsample_factor=100, overlap_ratio=0.5, normalization = None)
    return shift


def _applyRegisterMatrix(dst:np.ndarray, shift):
    
    t_m = np.array([[0,1,shift[0]],[1,0,shift[1]]])
    aligned = cv2.warpAffine(dst, t_m, dst.shape, borderValue=65535).T
    
    return aligned


def _EulerTransform(fix: sitk.SimpleITK.Image, move: sitk.SimpleITK.Image):
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fix)
    elastixImageFilter.SetMovingImage(move)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("rigid", numberOfResolutions=6))
    elastixImageFilter.Execute()
    
    transform_parameter_map = elastixImageFilter.GetTransformParameterMap()
    
    return transform_parameter_map


def _applyEulerTransform2048(transform_parameter_map, move: sitk.SimpleITK.Image):
    
    theta, dy, dx = transform_parameter_map[0]['TransformParameters']
    
    transform = sitk.Euler2DTransform()
    transform.SetCenter([1024,1024])
    transform.SetTranslation([float(dy), float(dx)])
    transform.SetAngle(float(theta))
    
    return sitk.Resample(move, transform, sitk.sitkLinear, 65535)

def _applyEulerTransform5000(transform_parameter_map, move: sitk.SimpleITK.Image):
    
    theta, dy, dx = transform_parameter_map[0]['TransformParameters']
    
    transform = sitk.Euler2DTransform()
    transform.SetCenter([2500, 2500])
    transform.SetTranslation([float(dy), float(dx)])
    transform.SetAngle(float(theta))
    
    return sitk.Resample(move, transform, sitk.sitkLinear, 65535)