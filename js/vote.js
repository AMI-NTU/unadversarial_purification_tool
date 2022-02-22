
export function vote_with_noises(model, denoised_model, img, num) {
  // Loss functions (of increasing complexity) that measure how close the image is to the target class
  let img_shape = img.shape
  let sigma = 0.25
  let noise = tf.randomNormal([num, img_shape[1], img_shape[2], img_shape[3]]).mul(tf.scalar(sigma))
  let img_noised_repeated = tf.tile(img, [num, 1, 1, 1]).add(noise).clipByValue(0, 1);
  let denoised_img = denoised_model.predict(img_noised_repeated)
  let pred = model.predict(denoised_img).reshape([num,10])
  let predLblIdx = pred.argMax(1).dataSync();
  let counts = count_vote(predLblIdx, 10)
  let lbl_major = counts.indexOf(Math.max(...counts));
  let count_major = Math.max(...counts)

  return [lbl_major, denoised_img, img_noised_repeated, denoised_img]
}

export function vote_with_masks(model, denoised_model, img, num) {
  // Loss functions (of increasing complexity) that measure how close the image is to the target class
  let img_shape = img.shape
  let mask_ratio = 0.5

  let uniform_noise = tf.randomUniform([num, img_shape[1], img_shape[2], 1])
  // let ones = tf.onesLike(uniform_noise)
  // let zeros = tf.zerosLike(uniform_noise)

  let condition = tf.onesLike(uniform_noise).mul(tf.scalar(mask_ratio))
  let mask = uniform_noise.greater(condition)

  let img_masked_repeated = tf.tile(img, [num, 1, 1, 1]).mul(mask);
  let demasked_img = denoised_model.predict(img_masked_repeated)
  let pred = model.predict(demasked_img).reshape([num,10])
  let predLblIdx = pred.argMax(1).dataSync();

  let counts = count_vote(predLblIdx, 10)
  let lbl_major = counts.indexOf(Math.max(...counts));
  let count_major = Math.max(...counts)
  return [lbl_major, count_major, img_masked_repeated, demasked_img]
}


export function count_vote(predLblIdx, num_classes) {
  let counts = new Array(num_classes).fill(0);
  for (const i of predLblIdx) {
    counts[i] = counts[i] + 1;
  }
  return counts
}
