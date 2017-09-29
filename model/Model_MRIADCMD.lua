-- @author Sachin Mehta


--- ----
-- Model that we used for WSI/ROI segmentation

local nn = require 'nn'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true

require 'nngraph'
local Dilated = nn.SpatialDilatedConvolution
local ReLU = cudnn.ReLU
local SBatchNorm = nn.SpatialBatchNormalization
local DeConvolution = cudnn.SpatialFullConvolution--cudnn
local Convolution = cudnn.SpatialConvolution
local classes = opt.classes

-- function to project the feature maps using 1x1 convolution
function projection(inF, outF, stride)
        local proj = nn.Sequential()
        proj:add(Convolution(inF, outF, 1, 1, stride, stride, 0,0))
        proj:add(SBatchNorm(outF))
        proj:add(ReLU(true))
        return proj
end

-- multi-scale context module with identity mappings using atrous or dilated convolutions
function context(n)

  local s2 = nn.Sequential()
  s2:add(Convolution(n,n,3,3,1,1,1,1))
  s2:add(SBatchNorm(n))
  s2:add(ReLU(true))

  local s = nn.Sequential()
  s:add(Dilated(n,n,3,3,1,1,2,2,2,2))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))

  local s1 = nn.Sequential()
  s1:add(Dilated(n,n,3,3,1,1,4,4,4,4))
  s1:add(SBatchNorm(n))
  s1:add(ReLU(true))

  local s3 = nn.Sequential()
  s3:add(Dilated(n,n,3,3,1,1,8,8,8,8))
  s3:add(SBatchNorm(n))
  s3:add(ReLU(true))

  local s4 = nn.Sequential()
  s4:add(Dilated(n,n,3,3,1,1,16,16,16,16))
  s4:add(SBatchNorm(n))
  s4:add(ReLU(true))

  local input = nn.Identity()()
  local b1 = s2(input)

  local b2 = s(b1)
  local res_1 = nn.CAddTable()({b2, b1, input})

  local b3 = s1(res_1)
  local res_2 = nn.CAddTable()({b3, b1, input})

  local b4 = s3(res_2)
  local res_3 = nn.CAddTable()({b4, b1, input})

  local b5 = s4(res_3)
  local res_4 = nn.CAddTable()({b5, b1, input})


  local contextAdd = nn.CAddTable()({b1, res_1, res_2, res_3, res_4, input})
  contextAdd = ReLU(true)(contextAdd)
  contextAdd = Convolution(n, n, 3, 3, 1, 1, 1, 1)(contextAdd)
  contextAdd = SBatchNorm(n)(contextAdd)
  contextAdd = ReLU(true)(contextAdd)

  local contextModel = nn.gModule({input}, {contextAdd})
  return contextModel
end

local function iadcmd(modelName)
  --variables at different CNN levels are named by assuming the input is of size 256
  -- this is just for the convenience

	local input = nn.Identity()()
	local model = torch.load(opt.encoder)
	local level1 = nn.Sequential()
	level1:add(model.modules[1])
	level1:add(model.modules[2])
	level1:add(model.modules[3])

  --projecting input to lower spatial resolution so that it can be combined with residual convolution units (RCU)
	local input_128 = cudnn.SpatialAveragePooling(3, 3, 2, 2, 1, 1)(input)
  local input_64 = cudnn.SpatialAveragePooling(3, 3, 2, 2, 1, 1)(input_128)

  local input_128_p = Convolution(3, 64, 1, 1, 1, 1, 0, 0)(input_128)
  input_128_p = Convolution(64, 64, 3, 3, 1, 1, 1, 1)(input_128_p)
  input_128_p = SBatchNorm(64)(input_128_p)
  input_128_p = ReLU(true)(input_128_p)

  local input_64_p = Convolution(3, 64, 1, 1, 1, 1, 0, 0)(input_64)
  input_64_p = Convolution(64, 64, 3, 3, 1, 1, 1, 1)(input_64_p)
  input_64_p = SBatchNorm(64)(input_64_p)
  input_64_p = ReLU(true)(input_64_p)



  local level0 = Convolution(3,classes,3,3,1,1,1,1)(input)
	level0 = SBatchNorm(classes)(level0)
	level0 = ReLU(true)(level0):annotate{name= modelName .. 'L0'}

	local L1_0 = level1(input)--:annotate{name= modelName .. 'L1'}
  local L1 = nn.CAddTable()({input_128_p, L1_0}):annotate{name= modelName .. 'L1'}
	
	local level2_0 = model.modules[4](L1)--:annotate{name= modelName .. 'level2'}
	local level2 = nn.CAddTable()({level2_0, input_64_p}):annotate{name= modelName .. 'level2'}

	local level3 = model.modules[5](level2):annotate{name= modelName .. 'level3'}
	local level4 = model.modules[6](level3):annotate{name= modelName .. 'level4'}
	local level5 = model.modules[7](level4):annotate{name= modelName .. 'level5'}
	local level6 = model.modules[8](level5):annotate{name= modelName .. 'level6'}


  --projections for dense connections between encoder and decoder
	--projection 112
	local proj_112_14 = projection(64, 256, 8)(L1)
  local proj_112_28 = projection(64, 128, 4)(L1)
  local proj_112_56 = projection(64, 64, 2)(L1)

	--projection 56
	local proj_56_14 = projection(64, 256, 4)(level2)
  local proj_56_28 = projection(64, 128, 2)(level2)

  local proj_56_14_ = projection(64, 256, 4)(level3)
  local proj_56_28_ = projection(64, 128, 2)(level3)

	--projection 28
	local proj_28_14 = projection(128, 256, 2)(level4)


  --dense decoder
	local level6Up = DeConvolution(512,256,3,3,2,2,1,1, 1, 1)(level6)
	level6Up = SBatchNorm(256)(level6Up)
	level6Up = ReLU(true)(level6Up):annotate{name= modelName .. 'level6Up'}

	local level56Add = nn.CAddTable()({level6Up, level5, proj_28_14, proj_56_14_, proj_112_14, proj_56_14})
	level56Add = DeConvolution(256,128,3,3,2,2,1,1, 1, 1)(level56Add)
	level56Add = SBatchNorm(128)(level56Add)
	level56Add = ReLU(true)(level56Add):annotate{name= modelName .. 'level56Add'}

	local level45Add = nn.CAddTable()({level56Add, level4, proj_56_28_, proj_56_28, proj_112_28})
	level45Add = DeConvolution(128,64,3,3,2,2,1,1, 1, 1)(level45Add)
	level45Add = SBatchNorm(64)(level45Add)
	level45Add = ReLU(true)(level45Add):annotate{name= modelName .. 'level45Add'}

	local level34Add = nn.CAddTable()({level45Add, level3, level2, proj_112_56})
	level34Add = DeConvolution(64,64,3,3,2,2,1,1, 1, 1)(level34Add)
	level34Add = SBatchNorm(64)(level34Add)
	level34Add = ReLU(true)(level34Add):annotate{name= modelName .. 'level34Add'}

	local level23Add = nn.CAddTable()({level34Add, L1})
	level23Add = DeConvolution(64,64,3,3,2,2,1,1, 1, 1)(level23Add)
	level23Add = SBatchNorm(64)(level23Add)
	level23Add = ReLU(true)(level23Add)
	level23Add = Convolution(64,classes,1,1,1,1,0,0)(level23Add)
	level23Add = SBatchNorm(classes)(level23Add)
	level23Add = ReLU(true)(level23Add):annotate{name= modelName .. 'level23Add'}


  --sparse decoder
  local hyper_1 = DeConvolution(512,classes,1,1,2,2, 0, 0,1,1)(level6):annotate{name= modelName .. 'hyper_level6'}
  local hyper_2 = Convolution(256,classes,1,1,1,1,0, 0)(level5):annotate{name= modelName .. 'hyper_level5'}
  local hyper_3 = nn.CAddTable()({hyper_1, hyper_2})--:annotate{name= modelName .. 'hyper_1_2_fuse'}
  hyper_3 = DeConvolution(classes,classes,1,1,2,2, 0, 0,1,1)(hyper_3)
  hyper_3 = SBatchNorm(classes)(hyper_3)
  hyper_3 = ReLU(true)(hyper_3):annotate{name= modelName .. 'hyper_l56_fuse'}
  local hyper_4 = Convolution(128,classes,1,1,1,1,0, 0)(level4):annotate{name= modelName .. 'hyper_level4'}
  hyper_3 = nn.CAddTable()({hyper_3, hyper_4})
  hyper_3 = DeConvolution(classes,classes,1,1,2,2,0, 0, 1,1)(hyper_3)
  hyper_3 = SBatchNorm(classes)(hyper_3)
  hyper_3 = ReLU(true)(hyper_3):annotate{name= modelName .. 'hyper_l564_fuse'}
  local hyper_5 = Convolution(64,classes,1,1,1,1,0, 0)(level3):annotate{name= modelName .. 'hyper_level3'}
  local hyper_6 = Convolution(64,classes,1,1,1,1,0, 0)(level2):annotate{name= modelName .. 'hyper_level2'}
  hyper_3 = nn.CAddTable()({hyper_3, hyper_5, hyper_6})
  hyper_3 = DeConvolution(classes,classes,1,1,2,2, 0, 0, 1,1)(hyper_3)
  hyper_3 = SBatchNorm(classes)(hyper_3)
  hyper_3 = ReLU(true)(hyper_3):annotate{name= modelName .. 'hyper_l56432_fuse'}

  local hyper_7 = Convolution(64,classes,1,1,1,1,0, 0)(L1):annotate{name= modelName .. 'hyper_L1'}
  hyper_3 = nn.CAddTable()({hyper_3, hyper_7})
  hyper_3 = DeConvolution(classes,classes,1,1,2,2, 0, 0, 1,1)(hyper_3)
  hyper_3 = SBatchNorm(classes)(hyper_3)
  hyper_3 = ReLU(true)(hyper_3):annotate{name= modelName .. 'hyper_lAll_fuse'}


	local level0Add = nn.CAddTable()({level23Add, level0, hyper_3})
	level0Add = Convolution(classes,classes,3,3,1,1,1,1)(level0Add)
	level0Add = SBatchNorm(classes)(level0Add)
	level0Add = ReLU(true)(level0Add):annotate{name= modelName .. 'level0Add'}
	
	local modelSeg = nn.gModule({input}, {level0Add}) 
	return modelSeg
end

--receives the 384x384 patch as input
local input_384_ = nn.Identity()()
local input_384 = SBatchNorm(3)(input_384_)

local start_dim = opt.cropStart --64
local end_dim = opt.cropEnd --256

--crop the 384x384 patch to 256 x 256
local input_256 = nn.Narrow(3, start_dim, end_dim)(input_384)
input_256 = nn.Narrow(4, start_dim, end_dim)(input_256)

--aggregate features at multiple resolutions
local model_384_out = iadcmd('s384_')(input_384)
local model_256_out = iadcmd('s256_')(input_256)

--crop the higher resolution patch
local crop_384 = nn.Narrow(3, start_dim, end_dim)(model_384_out)
crop_384 = nn.Narrow(4, start_dim, end_dim)(crop_384)

--fuse the feature maps at different resolutions
local fuse = nn.CAddTable()({crop_384, model_256_out})
local cont = context(classes)(fuse):annotate{name= 'context'}

local classifier = Convolution(classes,classes,3,3,1,1,1,1)(cont)
classifier = SBatchNorm(classes)(classifier)
classifier = ReLU(true)(classifier)
classifier = Convolution(classes,classes,1,1,1,1,0,0)(classifier)

local modelSeg = nn.gModule({input_384_}, {classifier}) 
modelSeg:cuda()


return modelSeg
