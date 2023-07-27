import torch
import torch.nn as nn
import torchvision.models as models

class base_model(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.num_classes = args.num_classes
        self.K = args.K
        self.bs = args.batch_size

        self.downstream_encoder = models.resnet50()
        self.downstream_encoder.fc = nn.Identity()

        self.pretrain_encoder = models.resnet50()
        self.pretrain_encoder.fc = nn.Identity()

        self.cls_head = nn.Linear(2048, self.num_classes)
        self.cls_head.weight.data.normal_(mean=0.0, std=0.01)

        # create the queue
        self.register_buffer("down_feat_queue", torch.rand(self.K, 2048, requires_grad=False))
        self.register_buffer("pre_feat_queue", torch.rand(self.K, 2048, requires_grad=False))
        self.register_buffer("label_queue", torch.ones(self.K, dtype=torch.int64, requires_grad=False))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long, requires_grad=False))
    
    def forward(self, x, label, i):
        down_feat = self.downstream_encoder(x)

        self.pretrain_encoder.eval()
        with torch.no_grad():
            pre_feat = self.pretrain_encoder(x)

        cali_pre_feat, cali_pre_label = self.semantic_calibration(down_feat, i)
    
        logit = self.cls_head(torch.cat((down_feat, cali_pre_feat), dim=0)) 
        new_label = torch.cat((label, cali_pre_label), dim=0)

        self._dequeue_and_enqueue(down_feat, label, pre_feat)

        return logit, new_label

    @torch.no_grad()
    def _dequeue_and_enqueue(self, down_feat, label, pre_feat):

        batch_size = down_feat.shape[0]
        ptr = int(self.queue_ptr)

        self.down_feat_queue[ptr : ptr + batch_size, :] = down_feat
        self.pre_feat_queue[ptr : ptr + batch_size, :] = pre_feat

        self.label_queue[ptr : ptr + batch_size] = label

        ptr = (ptr + batch_size) % self.K

        self.queue_ptr[0] = ptr

    @torch.no_grad()    
    def semantic_calibration(self, down_feat, iter=None):
        #save down_feat norm for numerical stable
        l2_norm = down_feat.norm(dim=1).mean() # a scalsr

        cali_pre_feat = self.norm(self.pre_feat_queue)
        normed_down_feat_queue = self.norm(self.down_feat_queue)

        # for faster training, rotation is time-consuming
        if iter % 10 == 0:
            # do dataset-level rotation
            cali_pre_feat = self.rot_align(cali_pre_feat, normed_down_feat_queue)

        # do class-level translation
        for c in range(self.num_classes):
            class_idx = torch.nonzero(self.label_queue == int(c)).squeeze() # [x]
            c_pre_feat = cali_pre_feat[class_idx]  # [x, dim]
            c_down_feat = normed_down_feat_queue[class_idx] # [x, dim]

            # mean average for pretrained class center
            pre_feat_center = c_pre_feat.mean(dim=0)   # [dim]

            # confidence guided average for downstream class center
            class_prototype = self.cls_head.weight[c] # [dim]
            distance_metric = c_down_feat @ self.norm(class_prototype, dim=0).unsqueeze(1) # [x, 1]
            weight = torch.softmax(distance_metric, dim=0) # [x, 1]
            down_feat_center = torch.mul(c_down_feat, weight).sum(dim=0) # [dim]
            align_vector = down_feat_center - pre_feat_center            # [dim]

            # translate and update
            c_pre_feat = c_pre_feat + align_vector 
            cali_pre_feat[class_idx] = c_pre_feat

        # align scale for numerical stable
        cali_pre_feat = self.norm(cali_pre_feat) * l2_norm # [K, dim]
        
        return cali_pre_feat, self.label_queue

    @torch.no_grad()
    def norm(self, x, dim=1):
        return torch.nn.functional.normalize(x, dim=dim)

    @torch.no_grad()
    def rot_align(self, X, Y):
        X = X.T # [D, N]
        Y = Y.T
        D, N = X.shape
        X_m = X.mean(dim=1).unsqueeze(1) # [D, 1]
        Y_m = Y.mean(dim=1).unsqueeze(1) # [D, 1]
        X = X - X_m                      # move to 0
        Y = Y - Y_m                      # move to 0
        S = X @ Y.T # [D, D]
        U, sig, VT = torch.linalg.svd(S)
        VT.T[:, -1] = VT.T[:, -1] * torch.det(VT.T@U.T)
        R = VT.T @ U.T

        t = (Y - R @ X).mean(1).unsqueeze(1) # [D, 1]
        X = R @ X + t.repeat(1, N)        # move back (to Y)

        return X.T # [N, D]

    @torch.no_grad()
    def test_forward(self, x):
        feat = self.downstream_encoder(x)
        logit = self.cls_head(feat)
        
        return logit
    
    @torch.no_grad()
    def warm_up_forward(self, x, label):
        self.pretrain_encoder.eval()
        pre_feat = self.pretrain_encoder(x)

        self._dequeue_and_enqueue(pre_feat, label, pre_feat)
