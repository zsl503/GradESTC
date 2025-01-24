import math
from typing import  Tuple, Union
import numpy as np
import torch
from torch import Tensor
from typing import Dict
from scipy.sparse.linalg import svds

import numpy as np
from sklearn.linear_model import LinearRegression

class Predictor:
    def __init__(self):
        pass

    def predict(self, number:float) -> int:
        pass

class DefaultPredictor(Predictor):
    def __init__(self):
        pass

    def predict(self, number:float) -> int:
        return math.ceil(number*1.3) + 1

class FixedPredictor(Predictor):
    def __init__(self, fixed_value:int):
        self.fixed_value = fixed_value

    def predict(self, number:float) -> int:
        return self.fixed_value

class ConstrainedSlidingWindowPredictor(Predictor):
    def __init__(self, window_size=5, adjustment_factor=1.10):
        """
        Initializes a sliding window predictor with constraints.
        :param window_size: sliding window size
        :param adjustment_factor: factor used to adjust the predicted value
        """
        super().__init__()
        self.window_size = window_size
        self.adjustment_factor = adjustment_factor
        self.history = []
        self.model = LinearRegression()

    def predict(self, number: float) -> int:
        """
        Make predictions based on historical data to ensure that the predicted value is not lower than the actual value.
        :param number: current actual value
        :return: the predicted value for the next step
        """
        self.history.append(number)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        if len(self.history) < self.window_size:
            return math.ceil(number * self.adjustment_factor)
        X = np.arange(self.window_size).reshape(-1, 1) 
        y = np.array(self.history).reshape(-1, 1)

        self.model.fit(X, y)
        next_pred = self.model.predict([[self.window_size]])[0][0]
        print(f"next_pred: {next_pred}")
        next_pred = max(number, next_pred)
        return math.ceil(next_pred * self.adjustment_factor)

class Compressor:
    # Base class for compression and decompression
    def __init__(self, **kwargs):
        pass

    def compress(self, tensor:torch.Tensor):
        pass

    def uncompress(self, tensor:torch.Tensor):
        pass
    
    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        pass

    def update_basis_by_vector(self, vector:torch.Tensor):
        pass

class TopkCompressor(Compressor):
    '''
    TopkCompressor class, for lossy compression based on top-k percentages.
    When compressing, only the K% elements with the largest absolute values are retained, and the result is converted to the specified sparse matrix format;
    When decompressing, the sparse matrix is converted back to a dense matrix.
    '''

    def __init__(self, k_percent: float, sparse_format: str = 'csr', **kwargs):
        super().__init__(**kwargs)
        self.k_percent = k_percent 
        if sparse_format.lower() not in ['coo', 'csr']:
            raise ValueError("sparse_format must be one of 'coo' or 'csr'")
        self.sparse_format = sparse_format.lower()

    def compress(self, tensor: torch.Tensor) -> Tuple[Union[torch.Tensor, torch.sparse.Tensor], torch.Tensor]:
        '''
        Args:
            tensor: torch.Tensor, the tensor to be compressed
        Returns:
            compressed_tensor: torch.sparse.Tensor, compressed sparse tensor
            error: torch.Tensor, error tensor
        '''
        original_shape = tensor.shape
        num_elements = tensor.numel()
        
        k = int(num_elements * (self.k_percent / 100.0))
        
        flat_tensor = tensor.flatten()
        abs_tensor = torch.abs(flat_tensor)

        _, indices = torch.topk(abs_tensor, k)
        compressed_tensor = torch.zeros_like(flat_tensor)
        compressed_tensor[indices] = flat_tensor[indices]
        compressed_tensor = compressed_tensor.unsqueeze(0)

        if self.sparse_format == 'coo':
            compressed_sparse_tensor = compressed_tensor.to_sparse_coo()
        elif self.sparse_format == 'csr':
            compressed_sparse_tensor = compressed_tensor.to_sparse_csr()
        error = tensor - compressed_tensor.reshape(original_shape)
        return compressed_sparse_tensor, error

    def uncompress(self, sparse_tensor: torch.sparse.Tensor, shape: Tuple[int, ...] = None) -> torch.Tensor:
        '''
        Args:
            sparse_tensor: torch.sparse.Tensor, compressed sparse tensor
            shape: tuple, shape of the original tensor (if needed)
        Returns:
            torch.Tensor, decompressed dense tensor
        '''
        if sparse_tensor.layout == torch.sparse_coo or sparse_tensor.layout == torch.sparse_csr:
            dense_tensor = sparse_tensor.to_dense()
            dense_tensor = dense_tensor.squeeze(0)
        
        if shape is not None:
            dense_tensor = dense_tensor.reshape(shape)
        
        return dense_tensor

    def update_basis(self, update_dict: dict):
        pass

    def update_basis_by_vector(self, vector: torch.Tensor):
        return {}

class SVDCompressor(Compressor):
    '''
    SVDCompress class for SVD compression and decompression, used for SVDFed algorithm
    '''
    def __init__(self, L, R, use_scale=True, **kwargs):
        self.U:torch.Tensor = None
        self.L = L # Adjust alpha
        self.R = R # Error threshold
        self.use_scale = use_scale


    def update_basis(self, update_dict:Dict[int, torch.Tensor]):
        max_index = max(update_dict.keys())
        key = next(iter(update_dict))
        L = update_dict.get(key).shape[0]
        self.U = torch.zeros(L, max_index+1)
                    
        # key is the updated position, value is the updated value
        for k, v in update_dict.items():
            self.U[:,k] = v.clone().detach()

    def get_svd_error(self, vector_t, U):
        # Calculate the error of each gradient in grads in turn
        total_error = 0
        for i in range(vector_t.shape[1]):
            g = vector_t[:,i].squeeze()
            alpha = U.T @ g
            # tmp = U @ alpha
            # alpha = min(g.norm() / tmp.norm(), self.L) * alpha
            g_approx = U @ alpha
            error = g - g_approx
            total_error += error.norm()/g.norm()
        return total_error/vector_t.shape[1]
    
    def update_basis_by_vector(self, vector:torch.Tensor):
        '''
        Return update_dict
        '''
        update_dict = {}
        vector_t = vector.T
        U, S, _ = torch.linalg.svd(vector_t, full_matrices=False)

        k = 0
        l = 0
        r = len(S)
        while l < r:
            mid = (l + r) // 2
            t = self.get_svd_error(vector_t, U[:,:mid])
            if t < self.R:
                r = mid
            else:
                l = mid + 1
        k = l

        self.U = U[:,:k]
        print(f"Layer basis selected maximun: {k}, error: {t}, threshold: {self.R}")
        # update_dict 为全部U向量
        for i in range(k):
            update_dict[i] = self.U[:,i]
        return update_dict
    
    def compress(self, vector:torch.Tensor):
        '''
        Args:
            vector: torch.Tensor, compressed tensor
        Returns:
            a: torch.Tensor, projection of the tensor in the basis
            e: torch.Tensor, error of the compressed tensor
        '''
        vector_t = vector.flatten().unsqueeze(0).T
        if self.U is None:
            return vector, torch.zeros_like(vector)
        alpha = self.U.T @ vector_t
        _g = self.U @ alpha

        if self.use_scale:
            scale = min((torch.std(vector_t)  + 1e-8)/ (torch.std(_g)  + 1e-8), self.L)
        else:
            scale = 1

        alpha = scale*alpha
        e = vector_t - scale*_g
        return alpha, e.T.reshape(vector.shape)
    
    def uncompress(self, alpha:torch.Tensor, shape = None):
        if alpha.shape == shape:
            return alpha.clone().detach()
        elif shape is None:
            return (self.U @ alpha).T
        else:
            return (self.U @ alpha).T.reshape(shape)

class QuickSlideSVDCompressor(Compressor):
    USE_PREMUTE = False
    def __init__(self, K, max_D, L, use_torch = False, gap = None, u_dtype='float32', **kwargs):
        self.K = K # Number of columns of U
        self.max_D = max_D # Actively updated dimensions
        self.L = L # The length of the parameter slice
        self.U = None # basis U

        self.cur_D = max_D
        if u_dtype == 'float128':
            self.dtype = np.float128
        elif u_dtype == 'float64':
            self.dtype = np.float64
        elif u_dtype== 'float32':
            self.dtype = np.float32
        elif u_dtype == 'float16':
            self.dtype = np.float16
        else:
            raise ValueError(f"Unsupported dtype {u_dtype}")
        if isinstance(use_torch, bool):
            self.use_torch = use_torch
        elif isinstance(use_torch, str):
            self.use_torch = use_torch.lower() == 'true'
        else:
            raise ValueError(f"Unsupported use_torch {use_torch}")
        self.predictor:Predictor = DefaultPredictor()
        self.gap = gap
        self.epoch = 0

    def set_predictor(self, predictor:Predictor):
        self.predictor = predictor

    def update_basis(self, update_dict: dict):
        if self.U is None:
            assert len(update_dict) == self.K, f"First update_dict length must be {self.K}"
            max_index = max(update_dict.keys())
            self.U = np.zeros((self.L, max_index + 1), dtype=self.dtype)  # Initialize U as numpy array
        # If U is read-only, make it writable
        if self.U.flags.writeable is False:
            self.U = self.U.copy()
        # Key is the update index, value is the update value
        for k, v in update_dict.items():
            self.U[:, k] = v.copy()

    def update_D(self, actual_D):
        predict_D = self.predictor.predict(actual_D)
        if predict_D > self.max_D:
            self.cur_D = self.max_D
        elif predict_D < 2:
            self.cur_D = 2
        else:
            self.cur_D = predict_D

    def _torch_update(self, vector_t: torch.Tensor, update_threshold:float=0):
        # Select torch random svd, and convert U and other data to tensor before calculation
        # Determine whether it is on GPU, if not, convert to GPU
        if vector_t.device.type != 'cuda' and torch.cuda.is_available():
            vector_t = vector_t.cuda()
        
        if self.U is None:
            # Compute U using SVD decomposition
            U, S, V = torch.linalg.svd(vector_t, full_matrices=False)
            U = U[:, :self.K]
            self.U = U.cpu().numpy().astype(self.dtype)
            # update indedx
            update_index = range(self.K)
        elif self.cur_D > 0:
            # Reconstruct vector using U
            # Turn U to tensor
            U = torch.from_numpy(self.U.copy()).to(vector_t.device, dtype=vector_t.dtype)
            e = vector_t - U @ U.T @ vector_t
            U_e, S_e, V_e = torch.linalg.svd(e, full_matrices=False)
            U_e = U_e[:, :self.cur_D]
            U_K_e = torch.cat([U, U_e], dim=1)
            alpha = U_K_e.T @ vector_t
            contribution = torch.sum(abs(alpha), dim=1)  # Calculate the contribution of each orthogonal vector
            min_indices = torch.argsort(contribution)[:self.cur_D]  # Get the D smallest contributing indices
            min_indices_set = set(min_indices.tolist())
            wait_D_update_set = set(range(self.K, self.K + self.cur_D))
            sub_index = min_indices_set - wait_D_update_set
            add_index = wait_D_update_set - min_indices_set
            # Swap columns
            U_K_e[:, list(sub_index)] = U_K_e[:, list(add_index)]
            alpha[list(sub_index)] = alpha[list(add_index)]
            U_K = U_K_e[:, :self.K]
            if update_threshold > 0:
                alpha_2 = alpha[:self.K]
                e_2 = vector_t - U_K @ alpha_2
                # Check if error difference is below the threshold, and skip update if so
                if (e.norm() - e_2.norm()) / e.norm() < update_threshold:
                    return {}
            self.U = U_K.cpu().numpy().astype(self.dtype)
            # Return updated columns in dictionary form
            update_index = list(sub_index)
            self.update_D(len(sub_index))   
        else:
            raise ValueError("Invalid D value")
        
        update_dict = {}
        for i in update_index:
            update_dict[i] = self.U[:, i].copy()
        return update_dict

    def _numpy_update(self, vector_t: np.ndarray, update_threshold:float=0):
        if self.U is None:
            # Compute U using SVD decomposition
            U, S, V = svds(vector_t, k=self.K)
            self.U = U.copy().astype(self.dtype)
            # update indedx
            update_index = range(self.K)

        elif self.cur_D > 0:
            # Reconstruct vector using U
            e = vector_t - self.U @ self.U.T @ vector_t
            U_e, S_e, V_e = svds(e, k=self.cur_D)
            U_K_e = np.hstack([self.U, U_e.astype(self.dtype)])  # Concatenate the new basis vectors
            alpha = U_K_e.T @ vector_t
            
            contribution = np.sum(abs(alpha), axis=1)  # Calculate the contribution of each orthogonal vector
            min_indices = np.argsort(contribution)[:self.cur_D]  # Get the D smallest contributing indices
            min_indices_set = set(min_indices.tolist())

            wait_D_update_set = set(range(self.K, self.K + self.cur_D))
            sub_index = min_indices_set - wait_D_update_set
            add_index = wait_D_update_set - min_indices_set

            # Swap columns
            U_K_e[:, list(sub_index)] = U_K_e[:, list(add_index)]
            alpha[list(sub_index)] = alpha[list(add_index)]
            U_K = U_K_e[:, :self.K]
            if update_threshold > 0:
                alpha_2 = alpha[:self.K]
                e_2 = vector_t - U_K @ alpha_2
                # Check if error difference is below the threshold, and skip update if so
                if (np.linalg.norm(e) - np.linalg.norm(e_2)) / np.linalg.norm(e) < update_threshold:
                    return {}
            
            self.U = U_K.copy()
            # Return updated columns in dictionary form
            update_index = list(sub_index)
            self.update_D(len(sub_index))
        else:
            raise ValueError("Invalid D value")
        update_dict = {}
        for i in update_index:
            update_dict[i] = self.U[:, i].copy()
        return update_dict

    def update_basis_by_vector(self, vector:torch.Tensor, update_threshold:float=0):
        '''
        Return update_dict
        '''
        # update U by vector
        flatten_L = vector.numel()
        if flatten_L % self.L != 0:
            return {}
        vector = vector.reshape(-1, self.L)
        if self.K > vector.shape[0]:
            raise ValueError(f"K {self.K} must less than vector.shape[0] {vector.shape[0]}")
        # return self.__numpy_update(vector.T.cpu().numpy())
        if self.gap is not None and self.epoch % self.gap == 0:
            # 清空U
            self.U = None
            print(f"Clear U at compress epoch {self.epoch}")
        self.epoch += 1
        if self.use_torch:
            return self._torch_update(vector.T)
        else:
            return self._numpy_update(vector.T.cpu().numpy())
    
    def compress(self, vector:torch.Tensor):
        '''
        Args:
            vector: torch.Tensor, compressed tensor, if the last dimension of vector is not divisible by L, it returns itself, otherwise it returns the compressed tensor
        Returns:
            a: torch.Tensor, the projection of the tensor under the basis
            e: torch.Tensor, the error of the compressed tensor
        '''
        flatten_L = vector.numel()
        if flatten_L % self.L != 0:
            print(f"vector.numel() {flatten_L} can't divide L {self.L}. Return itself")
            return vector, torch.zeros_like(vector)
   
        if QuickSlideSVDCompressor.USE_PREMUTE:
            vector_t = vector.permute(0, 2, 3, 1).reshape(-1, self.L).T
            U = torch.from_numpy(self.U).to(vector.device, dtype=vector.dtype)
            alpha = U.T @ vector_t
            g = U @ alpha
            e = vector_t - g

            shape = (vector.shape[0], vector.shape[2], vector.shape[3], vector.shape[1])
            return alpha, e.T.reshape(shape).permute(0, 3, 1, 2)
        else:
            vector_t = vector.reshape(-1, self.L).T
            U = torch.from_numpy(self.U).to(vector.device, dtype=vector.dtype)
            alpha = U.T @ vector_t
            g = U @ alpha
            e = vector_t - g
            return alpha, e.T.reshape(vector.shape)

    def uncompress(self, alpha:torch.Tensor, shape = None):
        # If the dimension of a is exactly equal to shape, return directly
        if alpha.shape == shape:
            return alpha.clone().detach()
        elif shape is None:
            U = torch.from_numpy(self.U).to(alpha.device, dtype=alpha.dtype)
            return (U @ alpha).T
        else:
            if QuickSlideSVDCompressor.USE_PREMUTE:
                U = torch.from_numpy(self.U).to(alpha.device, dtype=alpha.dtype)
                shape = (shape[0], shape[2], shape[3], shape[1])
                return (U @ alpha).T.reshape(shape).permute(0, 3, 1, 2)
            else:
                U = torch.from_numpy(self.U).to(alpha.device, dtype=alpha.dtype)
                return (U @ alpha).T.reshape(shape)

class RondomSlideSVDCompressor(QuickSlideSVDCompressor):
    def __init__(self, *value, **kwargs):
        super().__init__(*value, **kwargs)
    
    def _torch_update(self, vector_t: torch.Tensor, update_threshold:float=0):
        # Select torch random svd, and convert U and other data to tensor before calculation
        # Determine whether it is on GPU, if not, convert to GPU
        if vector_t.device.type != 'cuda' and torch.cuda.is_available():
            vector_t = vector_t.cuda()
        
        if self.U is None:
            # Compute U using SVD decomposition
            U, S, V = torch.linalg.svd(vector_t, full_matrices=False)
            U = U[:, :self.K]
            self.U = U.cpu().numpy().astype(self.dtype)
            # update indedx
            update_index = range(self.K)
        elif self.cur_D > 0:
            # Reconstruct vector using U
            # Turn U to tensor
            U = torch.from_numpy(self.U.copy()).to(vector_t.device, dtype=vector_t.dtype)
            rand_vector = torch.randn_like(vector_t)
            e = rand_vector - U @ U.T @ rand_vector
            U_e, S_e, V_e = torch.linalg.svd(e, full_matrices=False)
            U_e = U_e[:, :self.cur_D]
            U_K_e = torch.cat([U, U_e], dim=1)
            alpha = U_K_e.T @ vector_t
            contribution = torch.sum(abs(alpha), dim=1)  # Calculate the contribution of each orthogonal vector
            min_indices = torch.argsort(contribution)[:self.cur_D]  # Get the D smallest contributing indices
            min_indices_set = set(min_indices.tolist())
            wait_D_update_set = set(range(self.K, self.K + self.cur_D))
            sub_index = min_indices_set - wait_D_update_set
            add_index = wait_D_update_set - min_indices_set
            # Swap columns
            U_K_e[:, list(sub_index)] = U_K_e[:, list(add_index)]
            alpha[list(sub_index)] = alpha[list(add_index)]
            U_K = U_K_e[:, :self.K]
            if update_threshold > 0:
                alpha_2 = alpha[:self.K]
                e_2 = vector_t - U_K @ alpha_2
                # Check if error difference is below the threshold, and skip update if so
                if (e.norm() - e_2.norm()) / e.norm() < update_threshold:
                    return {}
            self.U = U_K.cpu().numpy().astype(self.dtype)
            # Return updated columns in dictionary form
            update_index = list(sub_index)
            self.update_D(len(sub_index))   
        else:
            raise ValueError("Invalid D value")
        
        update_dict = {}
        for i in update_index:
            update_dict[i] = self.U[:, i].copy()
        return update_dict

    def _numpy_update(self, vector_t: np.ndarray, update_threshold:float=0):
        if self.U is None:
            # Compute U using SVD decomposition
            U, S, V = svds(vector_t, k=self.K)
            self.U = U.copy().astype(self.dtype)
            # update indedx
            update_index = range(self.K)

        elif self.cur_D > 0:
            # Reconstruct vector using U
            rand_vector = np.random.randn(*vector_t.shape)
            e = rand_vector - self.U @ self.U.T @ rand_vector
            U_e, S_e, V_e = svds(e, k=self.cur_D)
            U_K_e = np.hstack([self.U, U_e.astype(self.dtype)])  # Concatenate the new basis vectors
            alpha = U_K_e.T @ vector_t
            
            contribution = np.sum(abs(alpha), axis=1)  # Calculate the contribution of each orthogonal vector
            min_indices = np.argsort(contribution)[:self.cur_D]  # Get the D smallest contributing indices
            min_indices_set = set(min_indices.tolist())

            wait_D_update_set = set(range(self.K, self.K + self.cur_D))
            sub_index = min_indices_set - wait_D_update_set
            add_index = wait_D_update_set - min_indices_set

            # Swap columns
            U_K_e[:, list(sub_index)] = U_K_e[:, list(add_index)]
            alpha[list(sub_index)] = alpha[list(add_index)]
            U_K = U_K_e[:, :self.K]
            if update_threshold > 0:
                alpha_2 = alpha[:self.K]
                e_2 = vector_t - U_K @ alpha_2
                # Check if error difference is below the threshold, and skip update if so
                if (np.linalg.norm(e) - np.linalg.norm(e_2)) / np.linalg.norm(e) < update_threshold:
                    return {}
            
            self.U = U_K.copy()
            # Return updated columns in dictionary form
            update_index = list(sub_index)
            self.update_D(len(sub_index))
        else:
            raise ValueError("Invalid D value")
        update_dict = {}
        for i in update_index:
            update_dict[i] = self.U[:, i].copy()
        return update_dict

class AllSlideSVDCompressor(QuickSlideSVDCompressor):
    def __init__(self, *value, **kwargs):
        super().__init__(*value, **kwargs)
    
    def _torch_update(self, vector_t, update_threshold = 0):
        if vector_t.device.type != 'cuda' and torch.cuda.is_available():
            vector_t = vector_t.cuda()
        
        # Compute U using SVD decomposition
        U, S, V = torch.linalg.svd(vector_t, full_matrices=False)
        U = U[:, :self.K]
        self.U = U.cpu().numpy().astype(self.dtype)
        # update indedx
        update_index = range(self.K)
        update_dict = {}
        for i in update_index:
            update_dict[i] = self.U[:, i].copy()
        return update_dict
    
    def _numpy_update(self, vector_t, update_threshold = 0):
        # Compute U using SVD decomposition
        U, S, V = svds(vector_t, k=self.K)
        self.U = U.copy().astype(self.dtype)
        # update indedx
        update_index = range(self.K)
        update_dict = {}
        for i in update_index:
            update_dict[i] = self.U[:, i].copy()
        return update_dict

class SlideSVDCompressor(QuickSlideSVDCompressor):
    def __init__(self, *value, **kwargs):
        super().__init__(*value, **kwargs)
        self.set_predictor(FixedPredictor(self.max_D))

class CompressorCombin:
    def __init__(self, setting_dict:Dict[str, tuple], class_name='SlideSVDCompressor', device='cpu', **kwargs):
        '''
        CompressorCombin class, used to combine multiple Compress classes, providing compression and decompression functions for multi-layer parameters
        Args:
        setting_dict: key is the parameter name, value is a tuple (K, D, L), K is the number of columns of U, D is the dimension to be actively updated, and L is the length of the parameter slice
        '''
        if not isinstance(setting_dict, dict):
            raise ValueError("setting_dict must be a dict")
        
        compressor:Compressor = globals()[class_name]
        if compressor is None:
            raise ValueError(f"Unsupported compressor {class_name}")

        self.setting_dict = setting_dict
        self.compressor_dict:Dict[str, Compressor] = {}
        for key, value in setting_dict.items():
            if isinstance(value, list):
                value = tuple(value)
            self.compressor_dict[key] = compressor(*value, **kwargs, device=device)
        self.device = device

    def update_basis_by_vector(self, model_params:Dict[str, Tensor]):
        '''
        Update the base of all compressors through model_params
        Args:
            model_params: model parameter dictionary
        Returns:
            dict: update dictionary
        '''
        res = {}
        for key, value in model_params.items():
            if key not in self.compressor_dict:
                continue
            compressor = self.compressor_dict[key]
            res[key] = compressor.update_basis_by_vector(value)
        return res

    def compress(self, model_params:Dict[str, Tensor], can_update_basis_func=None, **kwargs) -> Tuple[Dict[str, Tensor], dict, Dict[str, Tensor]]:
        '''
        Compress all compressor parameters in combine
        Args:
            model_params: model parameter dictionary, if key is not in compressor_dict, it will not be compressed
            can_update_basis_func: function to update the basis function, return True or False
        Returns:
            combin_alpha: compressed alpha dictionary
            combin_update_dict: update dictionary
        '''
        combin_alpha = {}
        combin_update_dict = {}
        combin_error = {}
        for key, value in model_params.items():
            if key not in self.compressor_dict:
                combin_alpha[key] = value
                combin_update_dict[key] = {}
                combin_error[key] = torch.zeros_like(value)
                continue
            compressor = self.compressor_dict[key]
            if can_update_basis_func is not None:
                if can_update_basis_func(**kwargs):
                    combin_update_dict[key] = compressor.update_basis_by_vector(value)
                else:
                    combin_update_dict[key] = {}
            combin_alpha[key], combin_error[key] = compressor.compress(value)
        return combin_alpha, combin_update_dict, combin_error

    def uncompress(self, combin_alpha:Dict[str, Tensor], templete_model_params:Dict[str, Tensor]) -> Dict[str, Tensor]:
        '''
        Decompress according to combin_alpha. If key is not in compressor_dict, no decompression is required
        Args:
            combin_alpha: compressed alpha dictionary
            templete_model_params: parameter template used to specify the shape of decompressed parameters
        Returns:
            dict: decompressed model parameters
        '''
        res = {}
        for key, value in combin_alpha.items():
            if key not in self.compressor_dict:
                res[key] = value
            else:
                res[key] = self.compressor_dict[key].uncompress(value, templete_model_params[key].shape)
        return res
    
    def update(self, combin_update_dict:Dict[str, Dict[int, Tensor]]):
        '''
        Update the base of all compressors in combine
        Args:
            combin_update_dict: key is the parameter name, value is the update dictionary, the key of the update dictionary is the update position, and value is the updated value
        '''
        for key, value in combin_update_dict.items():
            if key not in self.compressor_dict:
                continue
            compressor = self.compressor_dict[key]
            compressor.update_basis(value)

class QSGDQuantizer:
    def __init__(self, num_levels=8):
        self.num_levels = num_levels
        if num_levels == 8:
            self.value = 127
            self.type = torch.int8
        elif num_levels == 16:
            self.value = 32767
            self.type = torch.int16
        elif num_levels == 32:
            self.value = 2147483647
            self.type = torch.int32
        else:
            raise ValueError("Unsupported num_levels")

    def quantize(self, tensor:torch.Tensor):
        norm = tensor.norm(p=2)
        scale = norm / self.value
        sign = tensor.sign()
        abs_tensor = tensor.abs()
        q = (abs_tensor / scale).floor()
        prob = (abs_tensor / scale) - q
        rand_tensor = torch.rand_like(prob)
        q += torch.where(rand_tensor < prob, torch.ones_like(q), torch.zeros_like(q))
        quantized_tensor = sign * q
        return quantized_tensor.to(self.type), scale
    
    def dequantize(self, quantized_tensor, scale):
        dequantized_tensor = quantized_tensor * scale
        return dequantized_tensor
