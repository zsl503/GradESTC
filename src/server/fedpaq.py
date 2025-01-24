from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from typing import Any

from src.server.fedavg import FedAvgServer
from src.client.fedpaq import FedPAQClient
from src.utils.tools import NestedNamespace
from src.utils.compressor_utils import QSGDQuantizer
from src.utils.my_utils import calculate_data_size




class FedPAQServer(FedAvgServer):
    @staticmethod
    def get_hyperparams(args_list=None) -> Namespace:
        parser = ArgumentParser()
        parser.add_argument("--compress_level", type=int)
        parser.add_argument("--compress_global", type=bool)
        parser.add_argument("--compress_layer", type=list, default=None)
        return parser.parse_args(args=args_list)

    def __init__(
        self,
        args: NestedNamespace,
        algo: str = "FedPAQ",
        unique_model=False,
        use_fedavg_client_cls=False,
        return_diff=True,
    ):
        super().__init__(args, algo, unique_model, use_fedavg_client_cls, return_diff)
    
        if self.args.fedpaq.compress_layer is None:
            self.logger.log("The compress_layer is not set, all layers will be compressed.")
            self.args.fedpaq.compress_layer = list(self.model.state_dict().keys())
        self.compress_layer = self.args.fedpaq.compress_layer
        self.compress_level = self.args.fedpaq.compress_level
        self.compress_global = self.args.fedpaq.compress_global
        if self.compress_global:
            raise NotImplementedError("The global compression is not implemented yet.")
        self.init_trainer(FedPAQClient)
        self.quantizer = QSGDQuantizer(self.args.fedpaq.compress_level)

    def train_one_round(self):
        """The function of indicating specific things FL method need to do (at server side) in each communication round."""
        selected_clients = sorted(self.selected_clients)
        
        public_model_byte = calculate_data_size(self.public_model_params, set_sparse=self.set_sparse,set_layout=self.set_layout)
        
        clients_package = self.trainer.train()

        for client_id in selected_clients:
            self.clients_comm_recv_bytes[client_id] += public_model_byte
            assert self.return_diff, "The return_diff must be True in FedPAQ."
            try:
                byte = calculate_data_size(clients_package[client_id]['model_params_diff']["tensors"], 
                                        set_sparse=self.set_sparse, 
                                        set_layout=self.set_layout)
                byte += calculate_data_size(clients_package[client_id]['model_params_diff']["scales"], 
                        set_sparse=self.set_sparse, 
                        set_layout=self.set_layout)
            except:
                print(clients_package[client_id]['model_params_diff'])
            self.clients_comm_send_bytes[client_id] += byte

        self.aggregate(clients_package)

    def unpack_client_model(self, packed_model):
        quantized_model = packed_model["tensors"]
        scale = packed_model["scales"]
        dequantized_model = {}

        for key in quantized_model.keys():
            if scale[key] == 0:
                dequantized_model[key] = quantized_model[key].clone().detach()
            else:
                dequantized_model[key] = self.quantizer.dequantize(quantized_model[key], scale[key])
        return dequantized_model

    def aggregate(self, clients_package: OrderedDict[int, dict[str, Any]]):
        for client_id, package in clients_package.items():
            package["model_params_diff"] = self.unpack_client_model(package["model_params_diff"])
        
        super().aggregate(clients_package)