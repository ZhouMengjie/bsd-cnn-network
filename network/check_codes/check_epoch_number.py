import torch   

# resnet18 jc
model_file = 'model_junction/resnet18_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 recall junction', checkpoint['epoch'])

model_file = 'model_junction/resnet18_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 accuracy junction', checkpoint['epoch'])

model_file = 'model_junction/resnet18_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 precision junction', checkpoint['epoch'])

model_file = 'model_junction/resnet18_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 loss junction', checkpoint['epoch'])

model_file = 'model_junction/resnet18_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 F1 junction', checkpoint['epoch'])

# resnet18 bd
model_file = 'model_gap/resnet18_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 recall gap', checkpoint['epoch'])

model_file = 'model_gap/resnet18_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 accuracy gap', checkpoint['epoch'])

model_file = 'model_gap/resnet18_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 precision gap', checkpoint['epoch'])

model_file = 'model_gap/resnet18_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 loss gap', checkpoint['epoch'])

model_file = 'model_gap/resnet18_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet18 F1 gap', checkpoint['epoch'])

# resnet50 jc
model_file = 'model_junction_resnet50/resnet50_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 recall junction', checkpoint['epoch'])

model_file = 'model_junction_resnet50/resnet50_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 accuracy junction', checkpoint['epoch'])

model_file = 'model_junction_resnet50/resnet50_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 precision junction', checkpoint['epoch'])

model_file = 'model_junction_resnet50/resnet50_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 loss junction', checkpoint['epoch'])

model_file = 'model_junction_resnet50/resnet50_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 F1 junction', checkpoint['epoch'])

# resnet50 bd
model_file = 'model_gap_resnet50/resnet50_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 recall gap', checkpoint['epoch'])

model_file = 'model_gap_resnet50/resnet50_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 accuracy gap', checkpoint['epoch'])

model_file = 'model_gap_resnet50/resnet50_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 precision gap', checkpoint['epoch'])

model_file = 'model_gap_resnet50/resnet50_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 loss gap', checkpoint['epoch'])

model_file = 'model_gap_resnet50/resnet50_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('resnet50 F1 gap', checkpoint['epoch'])

# densenet161 jc
model_file = 'model_junction_densenet161/densenet161_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 recall junction', checkpoint['epoch'])

model_file = 'model_junction_densenet161/densenet161_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 accuracy junction', checkpoint['epoch'])

model_file = 'model_junction_densenet161/densenet161_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 precision junction', checkpoint['epoch'])

model_file = 'model_junction_densenet161/densenet161_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 loss junction', checkpoint['epoch'])

model_file = 'model_junction_densenet161/densenet161_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 F1 junction', checkpoint['epoch'])

# densenet161 bd
model_file = 'model_gap_densenet161/densenet161_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 recall gap', checkpoint['epoch'])

model_file = 'model_gap_densenet161/densenet161_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 accuracy gap', checkpoint['epoch'])

model_file = 'model_gap_densenet161/densenet161_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 precision gap', checkpoint['epoch'])

model_file = 'model_gap_densenet161/densenet161_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 loss gap', checkpoint['epoch'])

model_file = 'model_gap_densenet161/densenet161_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('densenet161 F1 gap', checkpoint['epoch'])


# vgg jc
model_file = 'model_junction_vgg/vgg_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg recall junction', checkpoint['epoch'])

model_file = 'model_junction_vgg/vgg_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg accuracy junction', checkpoint['epoch'])

model_file = 'model_junction_vgg/vgg_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg precision junction', checkpoint['epoch'])

model_file = 'model_junction_vgg/vgg_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg loss junction', checkpoint['epoch'])

model_file = 'model_junction_vgg/vgg_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg F1 junction', checkpoint['epoch'])

# vgg bd
model_file = 'model_gap_vgg/vgg_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg recall gap', checkpoint['epoch'])

model_file = 'model_gap_vgg/vgg_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg accuracy gap', checkpoint['epoch'])

model_file = 'model_gap_vgg/vgg_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg precision gap', checkpoint['epoch'])

model_file = 'model_gap_vgg/vgg_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg loss gap', checkpoint['epoch'])

model_file = 'model_gap_vgg/vgg_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('vgg F1 gap', checkpoint['epoch'])


# googlenet jc
model_file = 'model_junction_googlenet/googlenet_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet recall junction', checkpoint['epoch'])

model_file = 'model_junction_googlenet/googlenet_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet accuracy junction', checkpoint['epoch'])

model_file = 'model_junction_googlenet/googlenet_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet precision junction', checkpoint['epoch'])

model_file = 'model_junction_googlenet/googlenet_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet loss junction', checkpoint['epoch'])

model_file = 'model_junction_googlenet/googlenet_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet F1 junction', checkpoint['epoch'])

# googlenet bd
model_file = 'model_gap_googlenet/googlenet_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet recall gap', checkpoint['epoch'])

model_file = 'model_gap_googlenet/googlenet_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet accuracy gap', checkpoint['epoch'])

model_file = 'model_gap_googlenet/googlenet_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet precision gap', checkpoint['epoch'])

model_file = 'model_gap_googlenet/googlenet_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet loss gap', checkpoint['epoch'])

model_file = 'model_gap_googlenet/googlenet_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('googlenet F1 gap', checkpoint['epoch'])


# alexnet jc
model_file = 'model_junction_alexnet/alexnet_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet recall junction', checkpoint['epoch'])

model_file = 'model_junction_alexnet/alexnet_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet accuracy junction', checkpoint['epoch'])

model_file = 'model_junction_alexnet/alexnet_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet precision junction', checkpoint['epoch'])

model_file = 'model_junction_alexnet/alexnet_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet loss junction', checkpoint['epoch'])

model_file = 'model_junction_alexnet/alexnet_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet F1 junction', checkpoint['epoch'])

# alexnet bd
model_file = 'model_gap_alexnet/alexnet_recall.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet recall gap', checkpoint['epoch'])

model_file = 'model_gap_alexnet/alexnet_accuracy.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet accuracy gap', checkpoint['epoch'])

model_file = 'model_gap_alexnet/alexnet_precision.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet precision gap', checkpoint['epoch'])

model_file = 'model_gap_alexnet/alexnet_loss.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet loss gap', checkpoint['epoch'])

model_file = 'model_gap_alexnet/alexnet_F1.pth.tar'
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage) # load to CPU
print('alexnet F1 gap', checkpoint['epoch'])
