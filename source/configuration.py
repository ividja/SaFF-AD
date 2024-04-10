from functools import reduce


class ModelConfiguration():
    def __init__(self, **kwargs):
        self.input_dims = kwargs.get('dimensions', None)
        self.train_mode = kwargs.get('mode', None)
        self.model = kwargs.get('model', None)
        self.supervision = kwargs.get('supervision', None)
        self.kwargs = kwargs

        try:
            self.hidden_dims = kwargs.get('hidden_dims')
        except:
            self.hidden_dims = None

        self.application = kwargs.get('application', None)
        if self.application == 'classification':
            self.labelling = True
        elif self.application == 'pretraining':
            self.labelling = False
        if self.supervision == 'unsupervised':
            self.labelling = False

    def get_model(self): 
        
        if self.model is not None:
            if self.train_mode == '1D':
                input_config = reduce(lambda x, y: x * y, self.input_dims[1:] if self.input_dims[0] > 3 else self.input_dims)

            elif self.train_mode == '2D':
                if self.input_dims[0] > 3:
                    input_config = 2 if self.labelling else 1
                else:
                    input_config = self.input_dims[0] + 1 if self.labelling else self.input_dims[0]

            elif self.train_mode == '3D':
                input_config = 2 if self.labelling else 1
            net = self.model(input_config, dims = self.hidden_dims, **self.kwargs)
        else:                            
            dims_length = len(self.input_dims)
            product_of_dims = reduce(lambda x, y: x*y, self.input_dims) if dims_length > 1 else None

            if dims_length in [1, 2, 3, 4] and self.train_mode in ['1D', '2D', '3D']:
                network_config = self.train_mode

                if self.train_mode == '1D':
                    input_config = product_of_dims
                elif self.train_mode == '2D':
                    input_config = self.input_dims[0] + 1 if self.labelling else self.input_dims[0]
                elif self.train_mode == '3D':
                    if dims_length == 3:
                        input_config = 1  
                    if dims_length == 4:
                        input_config = self.input_dims[0] + 1 if self.labelling else self.input_dims[0]
            else:
                return ValueError('Invalid input dimensions or train mode')

            if network_config == '1D':
                from models import FFA as model
                net = model(input_config, self.hidden_dims, **self.kwargs)

            if network_config == '2D':
                from models import CFFA as model
                net = model(input_config, self.hidden_dims, **self.kwargs)

        return net
