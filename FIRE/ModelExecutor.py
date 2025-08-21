from typing import List, Literal, Tuple
from e3nn.o3 import Irreps
from abc import abstractmethod, ABC
from Model import SEMPNN, SEConv, SETransformer, SEMPRelaxed
from Model.BalancedIrreps import BalancedIrreps
from Dataset.CrystalSO3Dataset import CrystalSO3Dataset
import json

class ModelExecutor(ABC):
    def setup_datasets(self, config: dict) -> Tuple[CrystalSO3Dataset, CrystalSO3Dataset]:
        """
        Sets up the dataset for training

        Returns:
            The training dataset and the testing dataset
        """
        with open(f"{self.default_data_path}/train/data.json") as train_data:
            train_data = json.load(train_data)
        with open(f"{self.default_data_path}/validate/data.json") as validation_data:
            if config["validation"]:
                test_data = json.load(validation_data)
            else:
                train_data.extend(json.load(validation_data))
                with open(f"{self.default_data_path}/test/data.json") as test_data:
                    test_data = json.load(test_data)
        with open(f"./Dataset/short_atom_init.json") as atom_init:
            atom_init = json.load(atom_init)
        return (self.dataset_class(train_data, atom_init, config["lmax_h"], radius=config["radius"], 
                                   max_neighbors=config["max_neighbors"], normalize=config["normalize"]), 
                self.dataset_class(test_data, atom_init, config["lmax_h"], radius=config["radius"], 
                                   max_neighbors=config["max_neighbors"], normalize=config["normalize"]))

    @property
    @abstractmethod
    def default_data_path(self) -> str:
        """
        The location where all the data for this runner is stored. i.e Dataset/elastic
        """
        raise NotImplementedError("Set default data path in subclasses.")
    
    @abstractmethod
    def dataset_class(self, data: List[dict], atom_init: dict, lmax_h: int, radius: float, max_neighbors: int, normalize: bool) -> CrystalSO3Dataset:
        """
        Abstract factory for the dataset class. Takes in same arguments as a CrystalSO3Dataset and applies
        to a specific class
        """
        raise NotImplementedError("Set up which dataset class you want to use in a subclass")

    @property
    @abstractmethod
    def output_irreps_lst(self) -> List[Irreps]:
        """
        Defines the model output
        """
        raise NotImplementedError("Must choose output in subclasses")
    
    def setup_model(self, config: dict, num_atom_feats: int):
        """
        Sets up the models to be run with the right output style

        Args:
            config: The config file to use when setting up the models
            num_atom_feats: The number of atom features to use.

        Returns:
            The requested model.
        """
        if "segnn" == config["model"]:
            self.model = self.setup_sempgnn(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                                hidden_features=config["hidden_features"], num_graph_layers=config["num_graph_layers"],
                                num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
        elif "seconv" in config["model"]:
            self.model = self.setup_seconv(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], hidden_features=config["hidden_features"], 
                                num_graph_layers=config["num_graph_layers"], num_prepool_layers=config["num_prepool_layers"], 
                                num_postpool_layers=config["num_postpool_layers"], conv_type=config["model"][config["model"].rfind("-") + 1:],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
        elif "setransformer" == config["model"]:
            self.model = self.setup_setransformer(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                                hidden_features=config["hidden_features"], num_graph_layers=config["num_graph_layers"],
                                num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], max_radius=config["radius"], 
                                pooling=config["pooling"])
        elif "relaxed" == config["model"]:
            print('Relaxed model')
            self.model = self.setup_relaxed(num_atom_feats=num_atom_feats, lmax_h=config["lmax_h"], 
                                hidden_features=config["hidden_features"], num_graph_layers=config["num_graph_layers"],
                                num_prepool_layers=config["num_prepool_layers"], num_postpool_layers=config["num_postpool_layers"],
                                output_irreps_lst=self.output_irreps_lst, norm=config["layer_norm"], pooling=config["pooling"])
        else:
            raise Exception("No such model of that type")
            
    def setup_irreps(self, num_atom_feats: int, lmax_h: int, hidden_features: int) -> Tuple[Irreps, Irreps, Irreps, Irreps, Irreps, Irreps, Irreps]:
        """
        Sets up the irreps for the model input, hidden layers, and output

        Args:
            num_atom_feats: The number of atom scalar features provided.
            lmax_h: The maximum l for the hidden layers
            hidden_features: The number of hidden features

        Returns all the required irreps
        """
        # Scalar one hot encoding
        input_irreps = Irreps([(num_atom_feats, (0, 1))]) 
        
        #input_irreps = Irreps("126x0e + 1x1e") #manually set irreps for 1x1e force vector 
        print('Input irrep type:', input_irreps)
        # Irreps from relative position between atoms based on spherical harmonics
        edge_attr_irreps = Irreps.spherical_harmonics(lmax_h)
        node_attr_irreps = Irreps.spherical_harmonics(lmax_h)
        # Distance is a singular scalar
        additional_message_irreps = Irreps("1x0e")
        # Create hidden irreps
        hidden_irreps = BalancedIrreps(lmax_h, hidden_features)
        return input_irreps, edge_attr_irreps, node_attr_irreps, additional_message_irreps, hidden_irreps

    def setup_sempgnn(self, num_atom_feats: int, lmax_h: int, hidden_features: int, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]) -> SEMPNN:
        """
        Sets up a SEMPN model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_features: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            pooling: How to pool the nodes together

        Returns the specified SEMPNN model
        """
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_features)

        return SEMPNN(input_irreps,
            hidden_irreps,
            output_irreps_lst,
            edge_attr_irreps,
            node_attr_irreps,
            num_graph_layers=num_graph_layers,
            num_prepool_layers=num_prepool_layers,
            num_postpool_layers=num_postpool_layers,
            additional_message_irreps=additional_message_irreps,
            norm=norm,
            pooling=pooling)

    def setup_seconv(self, num_atom_feats: int, lmax_h: int, hidden_features: int, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, 
                    conv_type: Literal["linear", "nonlinear"], output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]) -> SEConv:
        """
        Sets up a SEConv model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_features: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            conv_type: Either linear or nonlinear convolutions
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            pooling: How to pool the nodes together

        Returns the specified SEConv model.
        """
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_features)

        return SEConv(input_irreps,
            hidden_irreps,
            output_irreps_lst,
            edge_attr_irreps,
            node_attr_irreps,
            num_graph_layers=num_graph_layers,
            num_prepool_layers=num_prepool_layers,
            num_postpool_layers=num_postpool_layers,
            additional_message_irreps=additional_message_irreps,
            conv_type=conv_type,
            norm=norm,
            pooling=pooling)

    def setup_setransformer(self, num_atom_feats: int, lmax_h: int, hidden_features: int, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], max_radius: float, pooling: Literal["mean", "add", "max"]) -> SETransformer:
        """
        Sets up a SETransformer model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_features: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            max_radius: The maximum radius for cutoff function
            pooling: How to pool the nodes together

        Returns the SETransformer model as specified
        """
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_features)

        return SETransformer(input_irreps,
            hidden_irreps,
            output_irreps_lst,
            edge_attr_irreps,
            node_attr_irreps,
            num_graph_layers=num_graph_layers,
            num_prepool_layers=num_prepool_layers,
            num_postpool_layers=num_postpool_layers,
            additional_message_irreps=additional_message_irreps,
            norm=norm,
            max_radius=max_radius,
            pooling=pooling)
    
    def setup_relaxed(self, num_atom_feats: int, lmax_h: int, hidden_features: int, num_graph_layers: int, 
                    num_prepool_layers: int, num_postpool_layers: int, output_irreps_lst: List[Irreps],
                    norm: Literal["none", "batch"], pooling: Literal["mean", "add", "max"]):
        """
        Sets up a SEConv model based on the parameters provided.

        Args:
            num_atom_feats: How many atomic features will be provided
            lmax_h: The maximum l for the hidden layer
            hidden_features: How many features in the hidden layers
            num_graph_layers: How many hidden layers of graph interaction
            num_prepool_layers: How many linear layers on just node values
            num_postpool_layers: How many layers before final output after pooling
            conv_type: Either linear or nonlinear convolutions
            output_irreps_lst: The list of outputs of the model
            norm: The layer normalization the model may employ
            pooling: How to pool the nodes together

        Returns the specified SEConv model.
        """
        (input_irreps, 
        edge_attr_irreps, 
        node_attr_irreps, 
        additional_message_irreps, 
        hidden_irreps) = self.setup_irreps(num_atom_feats, lmax_h, hidden_features)

        return SEMPRelaxed(input_irreps,
            hidden_irreps,
            output_irreps_lst,
            edge_attr_irreps,
            node_attr_irreps,
            num_graph_layers=num_graph_layers,
            num_prepool_layers=num_prepool_layers,
            num_postpool_layers=num_postpool_layers,
            additional_message_irreps=additional_message_irreps,
            norm=norm,
            pooling=pooling)
