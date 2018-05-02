import sys
caffe_root = "/home/xurj/ZJY/caffe-master/"
sys.path.insert(0, caffe_root + "python")
import caffe

net = caffe.Net("./train_val.prototxt", "./bvlc_reference_caffenet.caffemodel", caffe.TEST)

print net.params

layer_list = ["fc6", "fc7"]

for layer in layer_list:
    dst = layer + "_2"
    net.params[dst][0].data[...] = net.params[layer][0].data[...]
    net.params[dst][1].data[...] = net.params[layer][1].data[...]

net.save("./train.caffemodel")

