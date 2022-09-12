import multiprocessing
import shutil

import cv2
from core import pathex
from core.cv2ex import *
from core.interact import interact as io
from core.joblib import Subprocessor
from LFDIMG import *
from facelib import FaceType, LandmarksProcessor


class FacesetResizerSubprocessor(Subprocessor):

    #override
    def __init__(self, image_paths, output_dirpath, image_size, face_type=None):
        self.image_paths = image_paths
        self.output_dirpath = output_dirpath
        self.image_size = image_size
        self.face_type = face_type
        self.result = []

        super().__init__('FacesetResizer', FacesetResizerSubprocessor.Cli, 600)

    #override
    def on_clients_initialized(self):
        io.progress_bar (None, len (self.image_paths))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        base_dict = {'output_dirpath':self.output_dirpath, 'image_size':self.image_size, 'face_type':self.face_type}

        for device_idx in range( min(8, multiprocessing.cpu_count()) ):
            client_dict = base_dict.copy()
            device_name = f'CPU #{device_idx}'
            client_dict['device_name'] = device_name
            yield device_name, {}, client_dict

    #override
    def get_data(self, host_dict):
        if len (self.image_paths) > 0:
            return self.image_paths.pop(0)

    #override
    def on_data_return (self, host_dict, data):
        self.image_paths.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        io.progress_bar_inc(1)
        if result[0] == 1:
            self.result +=[ (result[1], result[2]) ]

    #override
    def get_result(self):
        return self.result

    class Cli(Subprocessor.Cli):

        #override
        def on_initialize(self, client_dict):
            self.output_dirpath = client_dict['output_dirpath']
            self.image_size = client_dict['image_size']
            self.face_type = client_dict['face_type']
            self.log_info (f"Running on { client_dict['device_name'] }")

        #override
        def process_data(self, filepath):
            try:
                lfdimg = LFDIMG.load (filepath)
                if lfdimg is None or not lfdimg.has_data():
                    self.log_err (f"{filepath.name} is not a lfd image file")
                else:
                    img = cv2_imread(filepath)
                    h,w = img.shape[:2]
                    if h != w:
                        raise Exception(f'w != h in {filepath}')
                    
                    image_size = self.image_size
                    face_type = self.face_type
                    output_filepath = self.output_dirpath / filepath.name
                    
                    if face_type is not None:
                        lmrks = lfdimg.get_landmarks()
                        mat = LandmarksProcessor.get_transform_mat(lmrks, image_size, face_type)
                        
                        img = cv2.warpAffine(img, mat, (image_size, image_size), flags=cv2.INTER_LANCZOS4 )
                        img = np.clip(img, 0, 255).astype(np.uint8)
                        
                        cv2_imwrite ( str(output_filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )

                        lfd_dict = lfdimg.get_dict()
                        lfdimg = LFDIMG.load (output_filepath)
                        lfdimg.set_dict(lfd_dict)
                        
                        matna_mask = lfdimg.get_matna_mask()
                        if matna_mask is not None:
                            matna_res = 256
                            
                            matna_lmrks = lmrks.copy()
                            matna_lmrks *= (matna_res / w)
                            matna_mat = LandmarksProcessor.get_transform_mat(matna_lmrks, matna_res, face_type)
                            
                            matna_mask = cv2.warpAffine(matna_mask, matna_mat, (matna_res, matna_res), flags=cv2.INTER_LANCZOS4 )
                            matna_mask[matna_mask < 0.5] = 0
                            matna_mask[matna_mask >= 0.5] = 1

                            lfdimg.set_matna_mask(matna_mask)
                        
                        seg_ie_polys = lfdimg.get_seg_ie_polys()
                        
                        for poly in seg_ie_polys.get_polys():
                            poly_pts = poly.get_pts()
                            poly_pts = LandmarksProcessor.transform_points(poly_pts, mat)
                            poly.set_points(poly_pts)
                            
                        lfdimg.set_seg_ie_polys(seg_ie_polys)
                        
                        lmrks = LandmarksProcessor.transform_points(lmrks, mat)
                        lfdimg.set_landmarks(lmrks)
    
                        image_to_face_mat = lfdimg.get_image_to_face_mat()
                        if image_to_face_mat is not None:
                            image_to_face_mat = LandmarksProcessor.get_transform_mat ( lfdimg.get_source_landmarks(), image_size, face_type )
                            lfdimg.set_image_to_face_mat(image_to_face_mat)
                        lfdimg.set_face_type( FaceType.toString(face_type) )
                        lfdimg.save()
                        
                    else:
                        lfd_dict = lfdimg.get_dict()
                         
                        scale = w / image_size
                        
                        img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LANCZOS4)                    
                        
                        cv2_imwrite ( str(output_filepath), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100] )

                        lfdimg = LFDIMG.load (output_filepath)
                        lfdimg.set_dict(lfd_dict)
                        
                        lmrks = lfdimg.get_landmarks()                    
                        lmrks /= scale
                        lfdimg.set_landmarks(lmrks)
                        
                        seg_ie_polys = lfdimg.get_seg_ie_polys()
                        seg_ie_polys.mult_points( 1.0 / scale)
                        lfdimg.set_seg_ie_polys(seg_ie_polys)
                        
                        image_to_face_mat = lfdimg.get_image_to_face_mat()
    
                        if image_to_face_mat is not None:
                            face_type = FaceType.fromString ( lfdimg.get_face_type() )
                            image_to_face_mat = LandmarksProcessor.get_transform_mat ( lfdimg.get_source_landmarks(), image_size, face_type )
                            lfdimg.set_image_to_face_mat(image_to_face_mat)
                        lfdimg.save()

                    return (1, filepath, output_filepath)
            except:
                self.log_err (f"Exception occured while processing file {filepath}. Error: {traceback.format_exc()}")

            return (0, filepath, None)

def process_folder ( dirpath):
    
    image_size = io.input_int(f"New image size", 512, valid_range=[128,2048])
    
    face_type = io.input_str ("Change face type", 'same', ['h','mf','f','wf','head','same']).lower()
    if face_type == 'same':
        face_type = None
    else:
        face_type = {'h'  : FaceType.HALF,
                     'mf' : FaceType.MID_FULL,
                     'f'  : FaceType.FULL,
                     'wf' : FaceType.WHOLE_FACE,
                     'head' : FaceType.HEAD}[face_type]
                     

    output_dirpath = dirpath.parent / (dirpath.name + '_resized')
    output_dirpath.mkdir (exist_ok=True, parents=True)

    dirpath_parts = '/'.join( dirpath.parts[-2:])
    output_dirpath_parts = '/'.join( output_dirpath.parts[-2:] )
    io.log_info (f"Resizing faceset in {dirpath_parts}")
    io.log_info ( f"Processing to {output_dirpath_parts}")

    output_images_paths = pathex.get_image_paths(output_dirpath)
    if len(output_images_paths) > 0:
        for filename in output_images_paths:
            Path(filename).unlink()

    image_paths = [Path(x) for x in pathex.get_image_paths( dirpath )]
    result = FacesetResizerSubprocessor ( image_paths, output_dirpath, image_size, face_type).run()

    is_merge = io.input_bool (f"\r\nMerge {output_dirpath_parts} to {dirpath_parts} ?", True)
    if is_merge:
        io.log_info (f"Copying processed files to {dirpath_parts}")

        for (filepath, output_filepath) in result:
            try:
                shutil.copy (output_filepath, filepath)
            except:
                pass

        io.log_info (f"Removing {output_dirpath_parts}")
        shutil.rmtree(output_dirpath)
