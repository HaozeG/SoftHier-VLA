#
# Copyright (C) 2025 ETH Zurich and University of Bologna
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Author: Chi Zhang <chizhang@ethz.ch>

#      K         N            N
#   |-----|   |-----|      |-----|
# M |  X  | x |  W  | K => |  Z  | M
#   |-----|   |-----|      |-----|

class SummaGEMM:

    def __init__(self):

        #GEMM parameters
        self.m_valid = 50
        self.n_valid = 720
        self.k_valid = 720
        self.check_abs_tol = 0.02
        self.dtype                   = 'fp16'
        self.m_size                  = 64
        self.n_size                  = 768
        self.k_size                  = 768

        #Hyperparamters Settings
        ## [Tile ]: tile size for each cluster
        self.m_tile                  = 32
        self.n_tile                  = 192
        self.k_tile                  = 96
        ## [Scale]: How many clusters (x=scale, y=scale) are assigned one GEMM.
        ##          For a set of clusters (x=scale, y=scale) we would call it **Group**
        self.summa_scale_x           = 4
        self.summa_scale_y           = 2
        ## [Group]: How many summa group
        ##          Do we need to reduce all groups
        ##          Adress gaps between groups
        self.summa_group_number      = 8
        self.summa_group_reduce      = 1
        self.summa_group_splitk      = 1
        self.summa_group_splitn      = 0
        self.summa_group_gap_x       = 192
        self.summa_group_gap_w       = 147456
        self.summa_group_gap_z       = 0
        ## [Resha]: Reshape options on input and output
        self.resha_x_from_enable     = 0
        self.resha_z_to_enable       = 0
        self.resha_x_from_m          = 128
        self.resha_z_to_m            = 2048
