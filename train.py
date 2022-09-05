import torch
import cfg
import torch.optim as optim
from model import YOLO_v3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    non_max_suppression,
    cellboxes_to_boxes,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
    get_loader,
    check_class_accuracy,
    plot_image,
)
from loss import YoloLoss
import warnings
warnings.filterwarnings('ignore')
torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(cfg.DEVICE)
        y0, y1, y2 = (
            y[0].to(cfg.DEVICE),
            y[1].to(cfg.DEVICE),
            y[2].to(cfg.DEVICE)
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        mean_loss = sum(losses) / len(losses)
        loop.set_postfix(loss=mean_loss)


def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cellboxes_to_boxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, prob_threshold=thresh, box_format="midpoint",
        )
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)


def main():
    model = YOLO_v3(num_classes=cfg.NUM_CLASSES).to(cfg.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    loss_fn = YoloLoss()
    scaler = torch.cuda.amp.GradScaler()

    train_loader, test_loader, train_eval_loader = get_loader(
        train_csv_path=cfg.DATA_SET+'/train.csv',
        test_csv_path=cfg.DATA_SET+'/test.csv'
    )

    if cfg.LOAD_MODEL:
        load_checkpoint(cfg.CHECKPOINT_FILE, model, optimizer, cfg.LEARNING_RATE, cfg.WEIGHT_DECAY)

    scaled_anchors = (
        torch.tensor(cfg.ANCHORS) * torch.tensor(cfg.S).unsqueeze(1).unsqueeze(2).repeat(1, 3, 2)
    ).to(cfg.DEVICE)
    # print(optimizer.state_dict()['param_groups'])
    map_list = []

    for epoch in range(cfg.NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)

        if epoch > 0 and epoch % 3 == 0:
            check_class_accuracy(model, test_loader, threshold=cfg.CONF_THRESHOLD)
            pred_boxes, true_boxes = get_bboxes(
                test_loader,
                model,
                iou_threshold=cfg.NMS_IOU_THRESH,
                anchors=cfg.ANCHORS,
                threshold=cfg.CONF_THRESHOLD
            )
            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=cfg.MAP_IOU_THRESH,
                box_format='midpoint',
                num_classes=cfg.NUM_CLASSES
            )
            print(f'Epoch: {epoch+1} MAP: {mapval}')
            if cfg.SAVE_MODEL:
                if len(map_list) > 0 and map_list[-1] < mapval:
                    map_list.append(mapval)
                    save_checkpoint(model, optimizer, file_name='checkpoint.pth.tar')
                elif len(map_list) == 0:
                    map_list.append(mapval)
                    save_checkpoint(model, optimizer, file_name='checkpoint.pth.tar')
            model.train()


if __name__ == '__main__':
    main()
