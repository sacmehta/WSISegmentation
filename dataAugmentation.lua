-- @author Sachin Mehta

require 'image'

--- --------
-- Module to augment the data
-- @module dataAugmentation

local dataAug = {}

--- -----
-- Function to flip the image
-- @function [parent=#dataAugmentation] flip_h
-- @param #image im RGB image
-- @param #image im RGB image
-- @param #image label Label image
-- @return #image Returns the flipped rgb image
-- @return #image Returns the flipped label image
-- 
function dataAug.flip_h(im, label)
  -- flip the image
   im = image.hflip(im)
   label = image.hflip(label)
  return im, label
end

--- -----
-- Function to crop and scale the image
-- @function [parent=#dataAugmentation] cropScale
-- @param #image im RGB image
-- @param #image im RGB image
-- @param #image label Label image
-- @return #image Returns the cropped + scaled rgb image
-- @return #image Returns the cropped + scaled label image
-- 
function dataAug.cropScale(im, label)
  -- select the position of cropping randomly
  local pos = {'c', 'tl', 'tr', 'bl', 'br'}
  local rand_val =  torch.random(1, 5)

  -- how much to crop (this is decided randomly for each image and range is set to be between 20 and 50 pixels)
  local crop_a = torch.random(20, 50)

  -- crop the image
  im = image.crop(im, pos[rand_val], opt.imWidth - crop_a, opt.imHeight - crop_a)
--  im1 = image.crop(im1, pos[rand_val], opt.imWidth - crop_a, opt.imHeight - crop_a)
  label = image.crop(label, pos[rand_val], opt.imWidth - crop_a, opt.imHeight - crop_a)
  
  -- scale the image
  im = image.scale(im, opt.imWidth, opt.imHeight, 'bilinear')
--  im1 = image.scale(im1, opt.imWidth, opt.imHeight, 'bilinear')
  label = image.scale(label, opt.imWidth, opt.imHeight, 'simple')
  return im, label
end

--- -----
-- Function to crop, scale, and then flip the image
-- @function [parent=#dataAugmentation] cropScaleFlip
-- @param #image im RGB image
-- @param #image label Label image
-- @return #image Returns the cropped + scaled + flipped rgb image
-- @return #image Returns the cropped + scaled + flipped label image
-- 
function dataAug.cropScaleFlip(im, label)
  -- select the position of cropping randomly
  local pos = {'c', 'tl', 'tr', 'bl', 'br'}
  local rand_val =  torch.random(1, 5)
  
  -- how much to crop (this is decided randomly for each image and range is set to be between 20 and 50 pixels)
  local crop_a = torch.random(20, 50)

  -- crop the image
  im = image.crop(im, pos[rand_val], opt.imWidth - crop_a, opt.imHeight - crop_a)
---  im1 = image.crop(im1, pos[rand_val], opt.imWidth - crop_a, opt.imHeight - crop_a)
  label = image.crop(label, pos[rand_val], opt.imWidth - crop_a, opt.imHeight - crop_a)
  
  -- scale the image
  im = image.scale(im, opt.imWidth, opt.imHeight, 'bilinear')
---  im1 = image.scale(im1, opt.imWidth, opt.imHeight, 'bilinear')
  label = image.scale(label, opt.imWidth, opt.imHeight, 'simple')
  
  -- flip the image
  im = image.hflip(im)
---  im1 = image.hflip(im1)
  label = image.hflip(label)
  return im, label
end

--- ------
-- Function to translate an image
-- @function [parent=#dataAugmentation] translate
-- @param #image im RGB image
-- @param #image label Label image
-- @return #image Returns the translated rgb image
-- @return #image Returns the translated label image

function dataAug.translate(im, label)
  local rand_x =  torch.random(10, 50)
  local rand_y =  torch.random(10, 50)
  
  --translate the image
  im = image.translate(im, rand_x, rand_y)
--  im1 = image.translate(im1, rand_x, rand_y)
  label = image.translate(label, rand_x, rand_y)
  label[label:eq(0)] = 1 --fill with background as torch doesn't accept 0 for loss
  return im, label
end

--- -----
-- Function to flip the image vertically
-- @function [parent=#dataAugmentation] flip_v
-- @param #image im RGB image
-- @param #image label Label image
-- @return #image Returns the flipped rgb image
-- @return #image Returns the flipped label image
-- 
function dataAug.flip_v(im, label)
  -- flip the image
   im = image.vflip(im)
  -- im1 = image.vflip(im1)
   label = image.vflip(label)
  return im,  label
end

return dataAug
