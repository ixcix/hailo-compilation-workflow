import numpy as np
import torch



class Pre_Bev_w_Head(torch.nn.Module):
    """ This includes the 3D parts PillarFeatureExtractor+scatter.
    """
    def __init__(self, full_model):
        super().__init__()
        self.full_model = full_model           
    
    def forward(self, batch_dict):
        for cur_module in self.full_model.module_list[:2]:
            batch_dict = cur_module(batch_dict)
        
        return batch_dict


class Bev_w_Head(torch.nn.Module):
    """ Same as backbone_2d + head, but accepting spatial_features directly.
         Wraps the original module which it accepts in constructor, code is copied from orig forward().
    """
    def __init__(self, bb2d, head):
        super().__init__()
        self._bb2d=bb2d  # the model.backbone_2d
        self.head=head    # the model.dense_head
        
    def forward(self, spatial_features):
        #2D backbone forward:
        ups = []
        ret_dict = {}
        x = spatial_features
        # print('x.shape', x.shape)
        for i in range(len(self._bb2d.blocks)):
            x = self._bb2d.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self._bb2d.deblocks) > 0:
                ups.append(self._bb2d.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self._bb2d.deblocks) > len(self._bb2d.blocks):
            x = self._bb2d.deblocks[-1](x)
        
        spatial_features_2d = x
        
        # DENSE HEAD FORWARD:

        # (AnchorHeadSingle forward code) pointpillars_kitti:
        # cls_preds = self.dense.conv_cls(spatial_features_2d)
        # box_preds = self.dense.conv_box(spatial_features_2d)
        # dir_cls_preds = self.dense.conv_dir_cls(spatial_features_2d)
        # return (spatial_features_2d, cls_preds, box_preds, dir_cls_preds)

        # #centerpoint-pillar dense head forward (CenterHead forward code):
        # x = self.dense.shared_conv(spatial_features_2d)
        # pred_dicts = []
        # for head in self.dense.heads_list:
        #     pred_dicts.append(head(x))
        # return pred_dicts

        # ---------- AnchorHeadMulti (solo CNN) ----------
        if self.head.shared_conv is not None:
            spatial_features_2d = self.head.shared_conv(spatial_features_2d)

        outputs = []
        for rpn_head in self.head.rpn_heads:
            cls = rpn_head.conv_cls(spatial_features_2d)

            if rpn_head.separate_reg_config is None:
                box = rpn_head.conv_box(spatial_features_2d)
            else:
                box_list = []
                for name in rpn_head.conv_box_names:
                    box_list.append(rpn_head.conv_box[name](spatial_features_2d))
                box = torch.cat(box_list, dim=1)

            if rpn_head.conv_dir_cls is not None:
                dir_cls = rpn_head.conv_dir_cls(spatial_features_2d)
                outputs.append((cls, box, dir_cls))
            else:
                outputs.append((cls, box))

        return outputs


class Post_Bev_w_Head(torch.nn.Module):
    """ This includes the non-neural anchor-base box-decoding piece of dense head module ("generate_predicted_boxes"),
         as well as the model's postprocessing (using 3D NMS).
    """
    def __init__(self, pp_full_model):
        super().__init__()
        self.pp_full_model = pp_full_model           
    
    def forward(self, bev_out):
        
        spatial_features_2d, cls_preds, box_preds, dir_cls_preds = bev_out # self._hailo_model(spatial_features_hailoinp)

        # print(cls_preds.shape, type(cls_preds), box_preds.shape)
        cls_preds = torch.Tensor(cls_preds)
        box_preds = torch.Tensor(box_preds)
        dir_cls_preds = torch.Tensor(dir_cls_preds)
        data_dict = {'batch_size': 1}        
        data_dict['spatial_features_2d'] = torch.Tensor(spatial_features_2d)
        
        batch_cls_preds, batch_box_preds = self.pp_full_model.dense_head.generate_predicted_boxes(
            batch_size=data_dict['batch_size'], cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False
        
        # Here's the unavoidable cuda part:    
        cuda_data_dict = {k: (v.cuda() if type(v)==torch.Tensor else v) for k,v in data_dict.items()}
        # pred_dicts, recall_dicts = self.pp_full_model.post_processing(cuda_data_dict)
        pred_dicts = self.pp_full_model.post_processing(cuda_data_dict)

        return pred_dicts[0][0]  # Now returns a dictionary directly
    