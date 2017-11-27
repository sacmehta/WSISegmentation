-- @author Sachin Mehta

require 'optim'
require 'xlua'
require 'image'
dataAug = require 'dataAugmentation'


--- ---------------
-- @module Module to train the network
-- 

confClasses = {}
for i=1,opt.classes do
  confClasses[i] = i
end

-- confusion matrix for training data
confusionMatTrain = optim.ConfusionMatrix(confClasses)

--save the training error to files
local trainLogger = optim.Logger(paths.concat(opt.snap, 'error_train_' .. opt.resumeEpoch .. '.log'))

-- extract models parameters
parameters, gradParameters = model:getParameters()

--specify the optimizer
if opt.optimizer == 'adam' then
  optimState = {
    learningRate = opt.lr,
    momentum = opt.m,
    learningRateDecay = 1e-7,
    weightDecay = opt.w
  }
  optimMethod = optim.adam
elseif opt.optimizer == 'sgd' then
  optimState = {
   learningRate = opt.lr,
   momentum = opt.m,
   learningRateDecay = 1e-7,
   nesterov = true,
   dampening = 0.0,
   weightDecay = opt.w
  }
  optimMethod = optim.sgd
else
  print('Please add the optimizer in the train file.')
  os.exit()
end

-- tensors to store the batch data
local inputsTr = torch.Tensor(opt.batchSize, 3, opt.imHeight, opt.imWidth)
--local inputsTr1 = torch.Tensor(opt.batchSize, 2, opt.imHeight, opt.imWidth)
local targetsTr = torch.Tensor(opt.batchSize, opt.imHeight-128, opt.imWidth-128)

inputsTr = inputsTr:cuda()
targetsTr = targetsTr:cuda()
--inputsTr1 = inputsTr1:cuda()
--- ------


-- Fucntion to train the network
-- @function [parent=#train] train
-- @param #number epoch Epoch number
-- @param #table dataset table that contains the details about the dataset such as image/label location

local function train(epoch, dataset)
  
  local time = sys.clock()
  
  --set the model to training state
  model:training()
  
  -- decay the learning rate after x epochs by d
  if epoch % opt.de == 0 then
    optimState.learningRate = optimState.learningRate/opt.d
  end
  
  print('Training Epoch --> '.. epoch .. ' [LR = ' .. optimState.learningRate .. ']')
  
  -- check if we want to augment the data or not
  local repeatData = 1
  if opt.aug == true then
    -- THis value needs to be changed if we want to need more augmentaion
    repeatData = opt.augType
  end
  
  local epochTrainErr = 0
  
  for rep = 1, repeatData do 
    -- shuffle the data
    shuffle = torch.randperm(trainingSize)

    for t = 1,trainingSize,opt.batchSize do
      xlua.progress(t, trainingSize)
      local idx = 1
      for i = t, math.min(t+opt.batchSize-1, trainingSize) do
        -- load new sample
        local rgbImg1 = image.load(dataset.trainIm[shuffle[i]]):float()
				
	-- We learn the mean and STD using Batch Normalization.
	-- If you want to use fixed mean and std, uncomment the following lines and change the mean (0.5) and std (1) values
--        rgbImg1[1]:add(-0.5)
--        rgbImg1[2]:add(-0.5)
--        rgbImg1[3]:add(-0.5)

        --rgbImg1[1]:div(1)
        --rgbImg1[2]:div(1)
        --rgbImg1[3]:div(1)
        --scaling is not required for melanoma
        --rgbImg1 = image.scale(rgbImg1, opt.imWidth, opt.imHeight)
        

        --rgbImg2[1]:div(dataset.std1[1])
        --rgbImg2[2]:div(dataset.std1[2])
        --scaling is not required for melanoma
        --rgbImg2 = image.scale(rgbImg2, opt.imWidth, opt.imHeight)

        local lblImg = image.load(dataset.trainlbl[shuffle[i]], 1, 'byte'):float()
        lblImg:add(dataset.labelAddVal)
        lblImg[lblImg:eq(0)] = 1
        lblImg[lblImg:gt(opt.classes)] = 1 -- in pascal dataset we have labels with value 255. Map such labels to 1 (background class)

	local start_dim = opt.cropStart --64
	local end_dim = opt.cropEnd --256
        lblImg = lblImg:narrow(2, start_dim, end_dim)
	lblImg = lblImg:narrow(3, start_dim, end_dim)



        if rep%2 == 0 then
          -- horizontal flipping
          rgbImg1, lblImg = dataAug.flip_h(rgbImg1, lblImg)
        elseif rep%3 == 0 then
          --translation
          rgbImg1, lblImg = dataAug.cropScale(rgbImg1, lblImg)
        elseif rep%4 == 0 then
          --Cropping + Scaling
          rgbImg1, lblImg = dataAug.translate(rgbImg1, lblImg)
        elseif rep%5 == 0 then
          -- cropping + scaling + flipping
          rgbImg1, lblImg = dataAug.cropScaleFlip(rgbImg1, lblImg)
        elseif rep%6 == 0 then
          -- vertical flipping
          rgbImg1, lblImg = dataAug.flip_v(rgbImg1, lblImg)
        end

        inputsTr[idx] = rgbImg1
        targetsTr[idx] = lblImg
        idx = idx + 1
      end

      --- ---
      -- Function to do forward and backward computation
      -- @function [parent=#train] feval
      -- @param #tensor parameters Network parameters
      -- @return #float training error for the batch
      -- @return #tensor gradient parameters
      function feval(parameters)
        -- reset gradients
        gradParameters:zero()
        local output = model:forward(inputsTr)
        local err = criterion:forward(output, targetsTr)

        -- estimate df/dW
        local df_do = criterion:backward(output, targetsTr)
        model:backward(inputsTr, df_do)
        
        
        local _, pred = output:max(2)
        confusionMatTrain:batchAdd(pred:view(-1), targetsTr:view(-1))
        
        if t < 100 then
         print('Sanity check. Error: ' .. err)
        end

        epochTrainErr = epochTrainErr + err
        -- return f and df/dX
        return err, gradParameters
      end
      optimMethod(feval, parameters, optimState)
    end
  end

  time = (sys.clock() - time)/trainingSize
  epochTrainErr = (epochTrainErr / (trainingSize/opt.batchSize))/repeatData
  print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')  
  print('Training Error: ' .. epochTrainErr)
  
  trainLogger:add{['Training Error '] = epochTrainErr, 
                  ['Learning rate'] = optimState.learningRate, 
                  ['Epoch'] = epoch} 
 
  --save the model
  local filename = paths.concat(opt.snap, 'model-' .. epoch .. '.t7')
  print('saving model file: ' .. filename)
  if opt.dpt == true then
        torch.save(filename, model:get(1):clearState()) --remove the dpt layer
  else
        torch.save(filename, model:clearState())
  end
 
  collectgarbage()
  return model, criterion, confusionMatTrain
end

return train
