-- @author Sachin Mehta


--- ----
-- Module to parse command line arguments
-- @module opts



local opts = {}

--- --------------------------------------------------
-- A function to parse the command line arguments
-- @function [parent=#opts] parse
-- @param #string arg command line arguments
-- --------------------------------------------------

function opts.parse(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Multi-Scale Residual Network')
  cmd:text()
  cmd:text('Command-Line options:')
  --general options
  cmd:option('-seed', 0, 'random seed')
  --training related options
  cmd:option('-lr', 0.0005, 'Initial Learning Rate')
  cmd:option('-d', 1, 'learning rate decay factor')
  cmd:option('-de', 50, 'decay learning rate by x epochs')
  cmd:option('-w', 5e-4, 'weight decay')
  cmd:option('-optimizer', 'sgd', 'adam or sgd')
  cmd:option('-m', 0.9, 'momentum')
  cmd:option('-batchSize', 10, 'Batch Size')
  cmd:option('-resume', 'false' , 'resume training')
  cmd:option('-resumeEpoch', 1, 'epoch from where to resume training')
  cmd:option('-maxEpoch', 50, 'Maximum number of epochs')
  --network related options
  cmd:option('-optimize', 1, 'optimize the model')
  cmd:option('-disp', 1, 'Print model and generate graph')
  --data related options
  cmd:option('-snap', './results/', 'save the models here')
  cmd:option('-datapath', './data/mel', 'Path to the dataset')
  cmd:option('-dataset', 'mel', 'Which dataset (mel for melanoma)?')
  cmd:option('-dataCache', './cache/', 'Path to directory for caching data related properties')
  cmd:option('-imWidth', 384, 'Image Width (256 for melanoma)')
  cmd:option('-imHeight', 384, 'Image Height (256 for melanoma)')
  cmd:option('-cropStart', 64, 'location from where to crop')
  cmd:option('-cropEnd', 256, 'Length of the cropped patch')
  cmd:option('-aug', 'true' , 'Data augmentation')
  cmd:option('-augType', 2, 'Which augmentation? (2 for h-flip, 3 - h-flip + translate, 4 - h-flip + translate + cropScale, 5 - h-flip + translate + cropScaleFlip, 6 - h-flip + translate + cropScaleFlip + v-flip ')
  --model related options
  cmd:option('-model', 'model/Model_MRIADCMD.lua', 'Model File')
  cmd:option('-encoder', './model/resnet-18.t7', 'Classification network pretrained on the ImageNet. Please download the pretrained model from here: https://github.com/facebook/fb.resnet.torch')
  
  
  local opt = cmd:parse(arg or {})
  opt.resume = opt.resume ~= 'false'
  opt.aug = opt.aug ~= 'false'

  --check if snapshot directory exist or not. If not, then create it
  if not paths.dirp(opt.snap) and not paths.mkdir(opt.snap) then
    cmd:error('Error: Unable to create snapshot directory: '.. opt.snap .. '\n')
  end
  --check if data directory exist or not
  if not paths.dirp(opt.datapath) then
    cmd:error('Error: Data directory does not exist : '.. opt.datapath .. '\n')
  end

  return opt
end

return opts
