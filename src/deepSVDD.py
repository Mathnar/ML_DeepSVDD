import json
import torch
from networks.main import build_network, build_autoencoder
from trainer.deepSVDD_trainer import DeepSVDDTrainer
from trainer.ae_trainer import AETrainer


class DeepSVDD(object):
    def __init__(self, objective, v):
        self.objective = objective
        self.v = v
        self.R = 0.0
        self.c = None

        self.type = None
        self.net = None  # neural network \phi
        self.trainer = None
        self.optimizer_name = None
        self.ae_pretrainer = None
        self.ae_trainer = None
        self.optimizer = None
        self.results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
            'label_array': None,
            'score_array': None,
        }

    def build_network(self, net_name):
        self.type = net_name
        self.net = build_network(net_name)

    def train(self, dataset, optimizer, lr, n_epochs,
              lr_milestones, batch_size, weight_decay, device,
              n_jobs_dataloader):
        self.optimizer_name = optimizer
        self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.v, optimizer, lr=lr,
                                       n_epochs=n_epochs, lr_milestones=lr_milestones, batch_size=batch_size,
                                       weight_decay=weight_decay, device=device, n_jobs_dataloader=n_jobs_dataloader)
        self.net = self.trainer.train(dataset, self.net)
        self.R = float(self.trainer.R.cpu().data.numpy())  # get float
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get list
        self.results['train_time'] = self.trainer.train_time

    def init_network_weights_from_pretraining(self):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_pretrainer.state_dict()
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        net_dict.update(ae_net_dict)
        self.net.load_state_dict(net_dict)

    def pretrain(self, dataset, ae_optimizer, lr, n_epochs,
                 lr_milestones, batch_size, weight_decay, device,
                 n_jobs_dataloader):
        self.ae_pretrainer = build_autoencoder(self.type)
        self.optimizer = ae_optimizer
        self.ae_trainer = AETrainer(ae_optimizer, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_pretrainer = self.ae_trainer.train(dataset, self.ae_pretrainer)
        self.ae_trainer.test(dataset, self.ae_pretrainer)
        self.init_network_weights_from_pretraining()

    def test(self, dataset, device, n_jobs_dataloader):
        if self.trainer is None:
            self.trainer = DeepSVDDTrainer(self.objective, self.R, self.c, self.v,
                                           device=device, n_jobs_dataloader=n_jobs_dataloader)
        self.trainer.test(dataset, self.net)
        self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        self.results['label_array'] = self.trainer.label_array
        self.results['score_array'] = self.trainer.score_array

    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_pretrainer.state_dict() if save_ae else None

        torch.save({'R': self.R,
                    'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False):
        model_dict = torch.load(model_path)
        self.R = model_dict['R']
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        if load_ae:
            if self.ae_pretrainer is None:
                self.ae_pretrainer = build_autoencoder(self.type)
            self.ae_pretrainer.load_state_dict(model_dict['ae_net_dict'])

    def save_results(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
