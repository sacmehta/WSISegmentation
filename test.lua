-- @author Sachin Mehta

require 'optim'
require 'xlua'
require 'image'

confusionMatTest = optim.ConfusionMatrix(confClasses)

--save the training error to files
local valLogger = optim.Logger(paths.concat(opt.snap, 'error_test_' .. opt.resumeEpoch .. '.log'))

local inputs = torch.Tensor(1, 3, opt.imHeight, opt.imWidth)
local targets = torch.Tensor(1, opt.imHeight-128, opt.imWidth-128)

inputs = inputs:cuda()
targets = targets:cuda()

--- ----
-- Function to test the network
-- @function [parent=#test] test
-- @param #number epoch Epoch number
-- @param #table dataset Table containing the information about the dataset

local function test(epoch, dataset)
  local time = sys.clock()
  model:evaluate()
  
  valSize = table.getn(dataset.valIm)
  validationErr = 0
  for i = 1,valSize do
    xlua.progress(i, valSize)
    local rgbImg = image.load(dataset.valIm[i]):float()
    --rgbImg = image.scale(rgbImg, opt.imWidth, opt.imHeight)
    
    --rgbImg[1]:add(-0.5)
    --rgbImg[2]:add(-0.5)
    --rgbImg[3]:add(-0.5)

    --rgbImg[1]:div(dataset.std1[1])
    --rgbImg[2]:div(dataset.std1[2])
    --rgbImg[3]:div(dataset.std1[3])
    

    --rgbImg1[1]:div(dataset.std2[1])
    --rgbImg1[2]:div(dataset.std2[2])
    
    local lblImg = image.load(dataset.vallbl[i], 1, 'byte'):float()
    --lblImg = image.scale(lblImg, opt.imWidth, opt.imHeight,  'simple')
    lblImg:add(dataset.labelAddVal)
    lblImg[lblImg:eq(0)] = 1
--    lblImg[lblImg:eq(3)] = 2
--    lblImg[lblImg:eq(4)] = 3
--    lblImg[lblImg:eq(5)] = 3
--    lblImg[lblImg:eq(7)] = 3
--    lblImg[lblImg:eq(6)] = 4
--    lblImg[lblImg:eq(8)] = 5
    lblImg[lblImg:gt(opt.classes)] = 1

    local start_dim = 64
        local end_dim = 256
       lblImg = lblImg:narrow(2, start_dim, end_dim)
        lblImg = lblImg:narrow(3, start_dim, end_dim)
    
    inputs[1] = rgbImg
    targets[1] = lblImg
    
    local output = model:forward(inputs)
    local err = criterion:forward(output,targets)
    validationErr = validationErr + err
    
    local _, pred = output:max(2)
    confusionMatTest:batchAdd(pred:view(-1), targets:view(-1))
  end
  
  time = (sys.clock() - time)/valSize
  validationErr = validationErr / valSize
  print("==> time to test 1 sample = " .. (time*1000) .. 'ms')  
  print('Validation Error: ' .. validationErr)
    
  valLogger:add{['Validation Error '] = validationErr, 
                ['Epoch'] = epoch} 
  
  
  --save the model
  --save the confusion matrix
  local filenameCon = paths.concat(opt.snap, 'con-' .. epoch .. '.txt')
  print('saving confusion matrix: ' .. filenameCon)
  local fileCon = io.open(filenameCon, 'w')
  
  fileCon:write("--------------------------------------------------------------------------------\n")
  fileCon:write("Training:\n")
  fileCon:write("================================================================================\n")
  fileCon:write(tostring(confusionMatTrain))
  fileCon:write("\n--------------------------------------------------------------------------------\n")
  fileCon:write("Testing:\n")
  fileCon:write("================================================================================\n")
  fileCon:write(tostring(confusionMatTest))
  fileCon:write("\n--------------------------------------------------------------------------------")
  fileCon:close()
  
  print('\n')
  confusionMatTest:zero()
  confusionMatTrain:zero()
  collectgarbage()
end

return test
