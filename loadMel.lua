-- @author Sachin Mehta

--- ---
-- File to load Pascal Context Dataset and Cache it
-- @module loadPascalContext

require 'image'

-- load the training and test files
local trainFile = opt.datapath .. '/train.txt'
local valFile = opt.datapath .. '/test.txt'

local classesName = { 'Background', '1', '2', '3', '4', '5', '6', '7'}

               
local classes = #classesName

--- ----
-- Function to check if file exists or not
-- @function [parent=#loadPascalContext] check_file 
-- @param #string name File name
-- @return #boolean Boolean indicating whether file exists or not
-- 
local function check_file(name)
   local f=io.open(name,"r")
   if f~=nil then 
    io.close(f) 
    return true 
   else 
    return false 
   end
end

trainImFileList = {}
trainIm1FileList = {}
trainLblFileList = {}
labelAddVal = 0

-- compute the mean and standard deviation for training data
-- do it offline
--local mean1 = {0.72075905156214, 0.66621883843212, 0.74148164596736}
--local std1 = {0.27756112640812, 0.28396234426101, 0.26049176971491}
--
local mean1 = {
  0.76453436636411,
  0.70927286806379,
  0.77908841987267
}
local std1 = {
  0.17024265830468,
  0.19564359777311,
  0.15810968250769
}

--local mean2 = {0.4572846486672, 0.43745873801202, 0.40406985094371}
--local std2 = {0.26742718836252, 0.26436273357389, 0.27838534104728}

--local mean2 = {0.16032004955071, 0.15418645213278}
--local std2 = {0.13351542658835, 0.12055973806927}

local mean2 = {
  0.15197299610137,
  0.15640383657466
}
local std2 = {
  0.11836776727387,
  0.13015500535413
}



local histClasses = torch.Tensor(classes):fill(0)

--parse the training data
if not check_file(trainFile) then
  print('Training file does not exist: ' .. trainFile)
  os.exit()
else
  lineNo = 0
  for line in io.lines(trainFile) do
    local col1, col2, col3 = line:match("([^,]+),([^,]+),([^,]+)")
    trainImFileList[lineNo] =opt.datapath .. col1
    trainLblFileList[lineNo] = opt.datapath .. col2
    trainIm1FileList[lineNo] = opt.datapath .. col3
    
    local labelIm = image.load(trainLblFileList[lineNo], 1, 'byte')
    --scale the label image using simple interpolation
    labelIm = image.scale(labelIm, opt.imWidth, opt.imHeight, 'simple')
    labelIm:add(labelAddVal)
    labelIm[labelIm:eq(0)] = 1
--    labelIm[labelIm:eq(3)] = 2
--    labelIm[labelIm:eq(4)] = 3
--    labelIm[labelIm:eq(5)] = 3
--    labelIm[labelIm:eq(7)] = 3
--    labelIm[labelIm:eq(6)] = 4
--    labelIm[labelIm:eq(8)] = 5
    histClasses = histClasses + torch.histc(labelIm:float(), classes, 1, classes)
    assert(torch.max(labelIm) <= classes and torch.min(labelIm) > 0, 'Label values should be between 1 and number of classes: max ' .. torch.max(labelIm) .. ' min: ' .. torch.min(labelIm))
    
    lineNo = lineNo + 1
  end
  assert(table.getn(trainImFileList) == table.getn(trainLblFileList), 'Number of images and labels are not equal')
end


--parse the validation data
valImFileList = {}
valLblFileList = {}
valIm1FileList = {}
--
if not check_file(valFile) then
  print('Validation file does not exist: ' .. valFile)
  os.exit()
else
  lineNo = 0
  for line in io.lines(valFile) do
    local col1, col2, col3 = line:match("([^,]+),([^,]+),([^,]+)")
    valImFileList[lineNo] =opt.datapath .. col1
    valLblFileList[lineNo] = opt.datapath .. col2
    valIm1FileList[lineNo] = opt.datapath .. col3
    
    local labelIm = image.load(valLblFileList[lineNo], 1, 'byte')
    --scale the label image using simple interpolation
    --labelIm = image.scale(labelIm, opt.imWidth, opt.imHeight, 'simple')
    --labelIm:add(labelAddVal)
    --assert(torch.max(labelIm) <= classes and torch.min(labelIm) > 0, 'Label values should be between 1 and number of classes: max ' .. torch.max(labelIm) .. ' min: ' .. torch.min(labelIm))
    
    lineNo = lineNo + 1
  end
  assert(table.getn(valImFileList) == table.getn(valLblFileList), 'Number of images and labels are not equal')
end


local normHist = histClasses / histClasses:sum()
local classWeights = torch.Tensor(classes):fill(1)
for i = 1, classes do
   -- Ignore unlabeled and egoVehicle
   if i == 1 then
      classWeights[i] = 0
   end
   if histClasses[i] < 1 then
      print("Class " .. tostring(i) .. " not found")
      classWeights[i] = 0
   else
      classWeights[i] = 1 / (torch.log(1.02 + normHist[i]))
   end
end


--cache the training and validation data information
dataCache = {}
dataCache.trainIm = trainImFileList
dataCache.trainIm1 = trainIm1FileList
dataCache.trainlbl = trainLblFileList
dataCache.valIm = valImFileList
dataCache.valIm1 = valIm1FileList
dataCache.vallbl = valLblFileList
dataCache.mean1 = mean1
dataCache.std1 = std1
dataCache.mean2 = mean2
dataCache.std2 = std2
dataCache.classes = classes
dataCache.labelAddVal = labelAddVal
dataCache.classWeight = classWeights

if not paths.dirp(opt.cacheDir) and not paths.mkdir(opt.cacheDir) then
  cmd:error('Error: Unable to create a cache directory: '.. opt.cacheDir .. '\n')
end

--save the details about the dataset
torch.save(opt.dataCacheFileName, dataCache)
