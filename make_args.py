from utils import load_json_file

class Args:
    def __init__(self, config_path):
        self.config = load_json_file(config_path)
        self.data_path = self.config['data_path']
        self.img_height = self.config['img_height']
        self.img_width = self.config['img_width']
        self.channels = self.config['channels']
        self.num_workers = self.config['num_workers']
        self.batch_size = self.config['batch_size']
        self.n_residual_blocks = self.config['n_residual_blocks']
        self.epochs = self.config['epochs']
        self.lr_decay_epoch = self.config['lr_decay_epoch']
        self.lambda_cyc = self.config['lambda_cyc']
        self.lambda_id = self.config['lambda_id']
        self.model_save_path = self.config['model_save_path']
        self.save_per_epochs = self.config['save_per_epochs']
        self.multi_gpu_flag = self.config['multi_gpu_flag']
        self.num_workers = self.config['num_workers']
        self.port_num = self.config['port_num']
        self.b1 = self.config['b1']
        self.b2 = self.config['b2']
        self.lr = self.config['lr']