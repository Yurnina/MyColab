import json
import shutil
import traceback
from pathlib import Path

import numpy as np

from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.leras import nn
from LFDIMG import *
from facelib import MatNaNet, LandmarksProcessor, FaceType
import pickle

def apply_matna(input_path, model_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')

    if not model_path.exists():
        raise ValueError(f'{model_path} not found. Please ensure it exists.')
        
    face_type = None
    
    model_dat = model_path / 'MatNa_data.dat'
    if model_dat.exists():
        dat = pickle.loads( model_dat.read_bytes() )
        dat_options = dat.get('options', None)
        if dat_options is not None:
            face_type = dat_options.get('face_type', None)
        
        
        
    if face_type is None:
        face_type = io.input_str ("MatNa model face type", 'same', ['h','mf','f','wf','head','same'], help_message="Specify face type of trained MatNa model. For example if MatNa model trained as WF, but faceset is HEAD, specify WF to apply matna only on WF part of HEAD. Default is 'same'").lower()
        if face_type == 'same':
            face_type = None
    
    if face_type is not None:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]
                     
    io.log_info(f'Applying trained MatNa model to {input_path.name}/ folder.')

    device_config = nn.DeviceConfig.ask_choose_device(choose_only_one=True)
    nn.initialize(device_config)
        
    
    
    matna = MatNaNet(name='MatNa', 
                    load_weights=True,
                    weights_file_root=model_path,
                    data_format=nn.data_format,
                    raise_on_no_model_files=True)
    matna_res = matna.get_resolution()
              
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        lfdimg = LFDIMG.load(filepath)
        if lfdimg is None or not lfdimg.has_data():
            io.log_info(f'{filepath} is not a LFDIMG')
            continue
        
        img = cv2_imread(filepath).astype(np.float32) / 255.0
        h,w,c = img.shape
        
        img_face_type = FaceType.fromString( lfdimg.get_face_type() )
        if face_type is not None and img_face_type != face_type:
            lmrks = lfdimg.get_source_landmarks()
            
            fmat = LandmarksProcessor.get_transform_mat(lmrks, w, face_type)
            imat = LandmarksProcessor.get_transform_mat(lmrks, w, img_face_type)
            
            g_p = LandmarksProcessor.transform_points (np.float32([(0,0),(w,0),(0,w) ]), fmat, True)
            g_p2 = LandmarksProcessor.transform_points (g_p, imat)
            
            mat = cv2.getAffineTransform( g_p2, np.float32([(0,0),(w,0),(0,w) ]) )
            
            img = cv2.warpAffine(img, mat, (w, w), cv2.INTER_LANCZOS4)
            img = cv2.resize(img, (matna_res, matna_res), interpolation=cv2.INTER_LANCZOS4)
        else:
            if w != matna_res:
                img = cv2.resize( img, (matna_res,matna_res), interpolation=cv2.INTER_LANCZOS4 )    
                    
        if len(img.shape) == 2:
            img = img[...,None]            
    
        mask = matna.extract(img)
        
        if face_type is not None and img_face_type != face_type:
            mask = cv2.resize(mask, (w, w), interpolation=cv2.INTER_LANCZOS4)
            mask = cv2.warpAffine( mask, mat, (w,w), np.zeros( (h,w,c), dtype=np.float), cv2.WARP_INVERSE_MAP | cv2.INTER_LANCZOS4)
            mask = cv2.resize(mask, (matna_res, matna_res), interpolation=cv2.INTER_LANCZOS4)
        mask[mask < 0.5]=0
        mask[mask >= 0.5]=1    
        lfdimg.set_matna_mask(mask)
        lfdimg.save()


        
def fetch_matna(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    output_path = input_path.parent / (input_path.name + '_matna')
    output_path.mkdir(exist_ok=True, parents=True)
    
    io.log_info(f'Copying faces containing MatNa polygons to {output_path.name}/ folder.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    
    files_copied = []
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        lfdimg = LFDIMG.load(filepath)
        if lfdimg is None or not lfdimg.has_data():
            io.log_info(f'{filepath} is not a LFDIMG')
            continue
        
        ie_polys = lfdimg.get_seg_ie_polys()

        if ie_polys.has_polys():
            files_copied.append(filepath)
            shutil.copy ( str(filepath), str(output_path / filepath.name) )
    
    io.log_info(f'Files copied: {len(files_copied)}')
    
    is_delete = io.input_bool (f"\r\nDelete original files?", True)
    if is_delete:
        for filepath in files_copied:
            Path(filepath).unlink()
            
    
def remove_matna(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : APPLIED MATNA MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED MATNA MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : APPLIED MATNA MASKS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
                               
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        lfdimg = LFDIMG.load(filepath)
        if lfdimg is None or not lfdimg.has_data():
            io.log_info(f'{filepath} is not a LFDIMG')
            continue
        
        if lfdimg.has_matna_mask():
            lfdimg.set_matna_mask(None)
            lfdimg.save()
            files_processed += 1
    io.log_info(f'Files processed: {files_processed}')
    
def remove_matna_labels(input_path):
    if not input_path.exists():
        raise ValueError(f'{input_path} not found. Please ensure it exists.')
    
    io.log_info(f'Processing folder {input_path}')
    io.log_info('!!! WARNING : LABELED MATNA POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED MATNA POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.log_info('!!! WARNING : LABELED MATNA POLYGONS WILL BE REMOVED FROM THE FRAMES !!!')
    io.input_str('Press enter to continue.')
    
    images_paths = pathex.get_image_paths(input_path, return_Path_class=True)
    
    files_processed = 0
    for filepath in io.progress_bar_generator(images_paths, "Processing"):
        lfdimg = LFDIMG.load(filepath)
        if lfdimg is None or not lfdimg.has_data():
            io.log_info(f'{filepath} is not a LFDIMG')
            continue

        if lfdimg.has_seg_ie_polys():
            lfdimg.set_seg_ie_polys(None)
            lfdimg.save()            
            files_processed += 1
            
    io.log_info(f'Files processed: {files_processed}')