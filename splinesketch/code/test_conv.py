from stroke_test import omniglot_control_points, omniglot_img, test_decoder_conv

test_decoder_conv(omniglot_img, omniglot_control_points, 1000, alpha=0.001)
