-- @author Sachin Mehta

--- ------
-- Module to intialize and set-up the model
-- @module init
-- 

require 'cunn'
require 'cudnn'
require 'nngraph'
cudnn.benchmark = true
cudnn.fastest = true

local M = {}

--- -----
-- Function to set-up the model (takes care of model related functionalities such as loading the model file or resuming from the snapshot)
-- @function [parent=#init] modelSetup
-- @return #model returns the network
-- @return #criterion returns the criterian used to compute the loss
-- @return #number returns the epoch number (from where to start the training, 1 for scratch and x for resuming)
-- 
function M.modelSetup()
  local model
  local epochNo

  if opt.resume then
    local fileToLoad = paths.concat(opt.snap , 'model-' .. opt.resumeEpoch .. '.t7')
    print('Resuming training from epoch number ' .. opt.resumeEpoch .. ' with model file :' .. fileToLoad)
    model = torch.load(fileToLoad)
    --context = paths.dofile('./model/Context.lua')
    --model = nn.Sequential()
    --model:add(model1)
    --model:add(context)
    epochNo = opt.resumeEpoch + 1
  else
    print('Starting training with ' .. opt.model)
    model = paths.dofile(opt.model)
    epochNo = 1
  end
 
  model:cuda() 
  if opt.optimize then
    local optnet = require 'optnet'
    local generateGraph = require 'optnet.graphgen'
    local sample_input1 = torch.FloatTensor(opt.batchSize, 3, opt.imHeight, opt.imWidth):cuda()--type(torch.CudaTensor)
 --   local sample_input2 = torch.CudaTensor(1, 2, opt.imHeight, opt.imWidth):uniform()--type(torch.CudaTensor)
    optnet.optimizeMemory(model, sample_input1, {inplace = false, mode = 'training'})
    
    --graphOpts = {
      --displayProps =  {shape='ellipse',fontsize=14, style='solid'},
      --nodeData = function(oldData, tensor)
      --  return oldData .. '\n' .. 'Size: '.. tensor:numel()
     -- end
    --}
    
    --g = generateGraph(model, {sample_input1, sample_input2}, graphOpts)
    
    --graph.dot(g, opt.snap.. '/graph_vis',opt.snap.. '/graph_vis')
  end
  
  if opt.disp == 1 then
    local optnet = require 'optnet'
    local graphgen = require 'optnet.graphgen'
    local iterm = require 'iterm'
    require 'iterm.dot'
    model:cuda()
    local sample_input1 = torch.CudaTensor(1, 3, opt.imHeight, opt.imWidth):uniform()--type(torch.CudaTensor)
    --local sample_input2 = torch.CudaTensor(1, 2, opt.imHeight, opt.imWidth):uniform()
    iterm.dot(graphgen(model, sample_input1), opt.snap..'/graph.png')
    print '==> here is the model:'
    print(model)
    print '==> paramerters: '
    a_, b_ = model:getParameters()
    print(a_:size(1))
  end
  
  --data parallelism
  --check how many GPUs are available and run on those GPUs
  local gpu_list = {}
  for i = 1,cutorch.getDeviceCount() do 
    gpu_list[i] = i 
  end
  print('No. of GPUs available are ' .. table.getn(gpu_list))
  if table.getn(gpu_list) > 1 then
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark
  
      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpu_list)
         :threads(function()
            local cudnn = require 'cudnn'
            require 'nngraph'
            cudnn.fastest, cudnn.benchmark = true, true
         end)
      dpt.gradInput = nil
      opt.dpt = true
      model = dpt:cuda()
   end
   
  local criterion = cudnn.SpatialCrossEntropyCriterion(opt.classWeight):cuda()
  
  --local criterion = nn.ParallelCriterion():add(criterionSingle):add(criterionSingle):add(criterionSingle):cuda()
  return model, criterion, epochNo
end

return M
