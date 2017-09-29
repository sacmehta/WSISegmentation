-- @author Sachin Mehta


--- -----
-- Module to start training
-- @module main

require 'torch'
require 'paths'
require 'optim'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

--- parse the command line arguments
local opts = require 'opts'
opt = opts.parse(arg) --global variable (can be accessed across files)

opt.cacheDir = opt.dataCache .. '/' .. opt.dataset
opt.dataCacheFileName = opt.cacheDir .. '/data.t7'

if opt.dataset == 'mel' then
  if not paths.filep(opt.dataCacheFileName) then
    print('Loading Melanoma dataset from loadMel')
    require 'loadMel'
  else
    print('loading cached Melanoma file')
  end
  dataset = torch.load(opt.dataCacheFileName)
else
  print('Wrong dataset specified. Please check')
  print('Exiting')
  os.exit()
end

--number of classes in the dataset
opt.classes = dataset.classes
print(opt.classes)
opt.classWeight = dataset.classWeight

print(opt.classWeight)

print('Dataset has ' .. opt.classes .. ' classes including background')

-- load the model
local models = require 'init'
model, criterion, epochNo = models.modelSetup()

local train = require 'train'
local test = require 'test'

--test(-1, dataset)

-- start training and validation
for ep = epochNo, opt.maxEpoch do
  model, criterion, confusionMat = train(ep, dataset)
  test(ep, dataset)
  collectgarbage()
end
