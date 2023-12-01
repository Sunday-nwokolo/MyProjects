# -*- coding: utf-8 -*-
#  Copyright 2023 United Kingdom Research and Innovation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
# Franck P. Vidal (Science and Technology Facilities Council)


from cil.framework import AcquisitionGeometry #, AcquisitionData, ImageData, ImageGeometry, DataOrder
from cil.io.TIFF import TIFFStackReader

import numpy as np
import os
from pathlib import Path
import json # Load the JSON file
from tifffile import imread



def distancePointLine(point, line) -> float:

    A = point
    B = line[0]
    C = line[1]
    
    d = (C - B) / np.linalg.norm(C - B)
    v = A - B;
    
    t = np.dot(v, d);
    P = B + t * d;
    return np.linalg.norm(P - A);
    
    
def getUnitOfLength(aUnitOfLength: str) -> float:

    unit_of_length = 0.0;

    km  = 1000.0    / 0.001;
    hm  =  100.0    / 0.001;
    dam =   10.0    / 0.001;
    m   =    1.0    / 0.001;
    dm  =    0.1    / 0.001;
    cm  =    0.01   / 0.001;
    mm  =    0.001  / 0.001;
    um  =    1.0e-6 / 0.001;

    if ((aUnitOfLength == "kilometer") or (aUnitOfLength == "kilometre") or (aUnitOfLength == "km")):
        unit_of_length = km;
    elif ((aUnitOfLength == "hectometer") or (aUnitOfLength == "hectometre") or (aUnitOfLength == "hm")):
        unit_of_length = hm;
    elif ((aUnitOfLength == "decameter") or (aUnitOfLength == "decametre") or (aUnitOfLength == "dam")):
        unit_of_length = dam;
    elif ((aUnitOfLength == "meter") or (aUnitOfLength == "metre") or (aUnitOfLength == "m")):
        unit_of_length = m;
    elif ((aUnitOfLength == "decimeter") or (aUnitOfLength == "decimetre") or (aUnitOfLength == "dm")):
        unit_of_length = dm;
    elif ((aUnitOfLength == "centimeter") or (aUnitOfLength == "centimetre") or (aUnitOfLength == "cm")):
        unit_of_length = cm;
    elif ((aUnitOfLength == "millimeter") or (aUnitOfLength == "millimetre") or (aUnitOfLength == "mm")):
        unit_of_length = mm;
    elif ((aUnitOfLength == "micrometer") or (aUnitOfLength == "micrometre") or (aUnitOfLength == "um")):
        unit_of_length = um;
    else:
        raise ValueError("Unknown unit of length (" + aUnitOfLength + ")");

    return unit_of_length;




class JSON2gVXRDataReader(object):

    '''
    Create a reader for gVXR's JSON files
    
    Parameters
    ----------
    file_name: str
        file name to read

    normalise: bool, default=True
        normalises loaded projections by detector white level (I_0)

    fliplr: bool, default = False,
        flip projections in the left-right direction (about vertical axis)
    '''
    
    def __init__(self,
                 file_name: str=None,
                 normalise: bool=True,
                 fliplr: bool=False):

        # Initialise class attributes to None
        self.file_name = None
        self.normalise = normalise
        self.fliplr = fliplr
        self._ag = None # The acquisition geometry object
        self.tiff_directory_path = None

        # The file name is set
        if file_name is not None:

            # Initialise the instance
            self.set_up(file_name=file_name, normalise=normalise, fliplr=fliplr)


    def set_up(self,
               file_name: str=None,
               normalise: bool=True,
               fliplr: bool=False):

        '''Set up the reader
        
        Parameters
        ----------
        file_name: str
            file name to read

        normalise: bool, default=True
            normalises loaded projections by detector white level (I_0)

        fliplr: bool, default = False,
            flip projections in the left-right direction (about vertical axis)
        '''

        # Save the attributes
        self.file_name = file_name
        self.normalise = normalise
        self.fliplr = fliplr

        # Check a file name was provided
        if file_name is None:
            raise ValueError('Path to JSON file is required.')

        # Check if the file exists
        file_name = os.path.abspath(file_name)
        if not(os.path.isfile(file_name)):
            raise FileNotFoundError('{}'.format(file_name))

        # Check the file name without the path
        file_type = Path(file_name).suffix
        if file_type.lower() != ".json":
            raise TypeError('This reader can only process JSON files. Got {}'.format(file_type))

        # Load the JSON file
        with open(self.file_name) as f:
            self.gVXR_params = json.load(f)

        # Get the absolute path of the JSON file
        cmd = os.path.abspath(self.file_name)

        # Get the path where the projections are
        projection_path = self.gVXR_params["Scan"]["OutFolder"]

        # Is an absolute path?
        if projection_path[0] == "/":
            self.tiff_directory_path = Path(projection_path)

        # It is a relative path
        else:
            # Get the absolute path of the JSON file
            file_abs_path = os.path.abspath(self.file_name)
            file_path = os.path.dirname(file_abs_path)
            self.tiff_directory_path = Path(file_path + "/" + projection_path)
            
        # Look for projections
        if not os.path.isdir(self.tiff_directory_path):
            raise ValueError(f"The projection directory '{self.tiff_directory_path}' does not exist")

        # Get the number of projections
        number_of_projections = self.gVXR_params["Scan"]["NumberOfProjections"]

        # Look for the name of projection images (use either one or two 'f' in .tiff)
        image_file_names = [image for image in self.tiff_directory_path.rglob("*.tiff")]
        if len(image_file_names) == 0:
            image_file_names = [image for image in self.tiff_directory_path.rglob("*.tif")]

        if len(image_file_names) != number_of_projections:
            raise IOError("Expecting " + str(number_of_projections) + " TIFF files , " + str(len(image_file_names)) + " were found in " + str(self.tiff_directory_path))

        # Get the rotation parameters
        rotation_axis_direction = -np.array(self.gVXR_params["Detector"]["UpVector"])
        rotation_axis_direction[2] *= -1
        
        if "CenterOfRotation" in self.gVXR_params["Scan"]:
            temp_rotation_axis_position = self.gVXR_params["Scan"]["CenterOfRotation"]
        elif "CentreOfRotation" in self.gVXR_params["Scan"]:
            temp_rotation_axis_position = self.gVXR_params["Scan"]["CentreOfRotation"]
        else:
            temp_rotation_axis_position = [0, 0, 0, "mm"]

        if len(temp_rotation_axis_position) == 4:
            rotation_axis_position = np.array(temp_rotation_axis_position[0:3]) * getUnitOfLength(temp_rotation_axis_position[3]) / getUnitOfLength("mm")
        rotation_axis_position[2] *= -1
        
        # Get the source position in mm
        temp = self.gVXR_params["Source"]["Position"]
        source_position_mm = np.array(temp[0:3]) * getUnitOfLength(temp[3]) / getUnitOfLength("mm")
        source_position_mm[2] *= -1

        # Get the detector position in mm
        temp = self.gVXR_params["Detector"]["Position"]
        detector_position_mm = np.array(temp[0:3]) * getUnitOfLength(temp[3]) / getUnitOfLength("mm")
        detector_position_mm[2] *= -1

        # Compute the ray direction
        ray_direction = (detector_position_mm - source_position_mm)
        ray_direction /= np.linalg.norm(ray_direction)

        # Get the shape of the beam (parallel vs cone beam)
        source_shape = self.gVXR_params["Source"]["Shape"]

        # Is it a parallel beam
        use_parallel_beam = False
        if type(source_shape) == str:
            if source_shape.upper() == "PARALLELBEAM" or source_shape.upper() == "PARALLEL":
                use_parallel_beam = True

        # Get the pixel spacing in mm
        detector_number_of_pixels = self.gVXR_params["Detector"]["NumberOfPixels"]
        if "Spacing" in self.gVXR_params["Detector"].keys() == list and "Size" in self.gVXR_params["Detector"].keys():
            raise ValueError("Cannot use both 'Spacing' and 'Size' for the detector")

        if "Spacing" in self.gVXR_params["Detector"].keys():
            temp = self.gVXR_params["Detector"]["Spacing"]
            pixel_spacing_mm = [
                temp[0] * getUnitOfLength(temp[2]) / getUnitOfLength("mm"),
                temp[1] * getUnitOfLength(temp[2]) / getUnitOfLength("mm")
             ]

        elif "Size" in self.gVXR_params["Detector"].keys():
            detector_size = self.gVXR_params["Detector"]["Size"];
            pixel_spacing_mm = [
                (detector_size[0] / detector_number_of_pixels[0]) * getUnitOfLength(detector_size[2]) / getUnitOfLength("mm"),
                (detector_size[0] / detector_number_of_pixels[0]) * getUnitOfLength(detector_size[2]) / getUnitOfLength("mm")
            ]
        else:
            raise ValueError("'Spacing' and 'Size' were not defined for the detector, we cannot determined the pixel spacing")

        # Get the angles
        include_final_angle = False
        if "IncludeFinalAngle" in self.gVXR_params["Scan"]:
            include_final_angle = self.gVXR_params["Scan"]["IncludeFinalAngle"]

        angle_set = -np.linspace(
            0,
            self.gVXR_params["Scan"]["FinalAngle"],
            self.gVXR_params["Scan"]["NumberOfProjections"],
            include_final_angle
        )

        detector_direction_x = np.cross(ray_direction, rotation_axis_direction)
        detector_direction_y = rotation_axis_direction
    
        if "RightVector" in self.gVXR_params["Detector"]:
            detector_direction_x = np.array(self.gVXR_params["Detector"]["RightVector"]);
            detector_direction_x[2] *= -1

        # Parallel beam
        if use_parallel_beam:
            self._ag = AcquisitionGeometry.create_Parallel3D(ray_direction,
                detector_position_mm,
                detector_direction_x=detector_direction_x,
                detector_direction_y=detector_direction_y,
                rotation_axis_position=rotation_axis_position,
                rotation_axis_direction=rotation_axis_direction,
                units="mm")
            print(ray_direction)
            print(detector_position_mm)
            print(rotation_axis_position)
            print(rotation_axis_direction)
        # It is cone beam
        else:
            self._ag = AcquisitionGeometry.create_Cone3D(source_position_mm,
                detector_position_mm,
                detector_direction_x=detector_direction_x,
                detector_direction_y=detector_direction_y,
                rotation_axis_position=rotation_axis_position,
                rotation_axis_direction=rotation_axis_direction,
                units="mm")

        # Set the angles of rotation
        self._ag.set_angles(-angle_set)

        # Panel is width x height
        self._ag.set_panel(detector_number_of_pixels, pixel_spacing_mm)
        self._ag.set_labels(['angle','vertical','horizontal'])


    def read(self):
        
        '''
        Reads projections and returns AcquisitionData with corresponding geometry,
        arranged as ['angle', horizontal'] if a single slice is loaded
        and ['vertical, 'angle', horizontal'] if more than 1 slice is loaded.
        '''

        # Check a file name was provided
        if self.tiff_directory_path is None:
            raise ValueError('The reader was not set properly.')

        # Create the TIFF reader
        reader = TIFFStackReader()

        reader.set_up(file_name=self.tiff_directory_path)

        ad = reader.read_as_AcquisitionData(self._ag)

        flat_field_correction_already_done = bool(self.gVXR_params["Scan"]["Flat-Field Correction"]) if "Flat-Field Correction" in self.gVXR_params["Scan"] else False

        if (self.normalise and not flat_field_correction_already_done):
            white_level = np.max(ad.array)
            ad.array[ad.array < 1] = 1

            # cast the data read to float32
            ad = ad / np.float32(white_level)
            
        
        if self.fliplr:
            dim = ad.get_dimension_axis('horizontal')
            ad.array = np.flip(ad.array, dim)
        
        return ad

    def load_projections(self):
        '''alias of read for backward compatibility'''
        return self.read()


    def get_geometry(self):
        
        '''
        Return AcquisitionGeometry object
        '''
        
        return self._ag

    def get_geometry(self):
        '''
        Return the acquisition geometry object
        '''
        return self._ag


