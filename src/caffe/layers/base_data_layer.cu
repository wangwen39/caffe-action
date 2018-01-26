#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
  }
  if (this->output_overlaps_) {
    // Reshape to loaded overlaps.
    top[2]->ReshapeLike(prefetch_current_->overlap_);
    top[2]->set_gpu_data(prefetch_current_->overlap_.mutable_gpu_data());
  }
}

template <typename Dtype>
void ReidPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // CHECK
  CHECK_EQ(top[0]->count(), prefetch_current_->data_.count()*2);
  // Reshape to loaded data.
  vector<int> data_shape = prefetch_current_->data_.shape();
  data_shape[0] *= 2;
  top[0]->Reshape(data_shape);
  // Copy the data
  caffe_copy(prefetch_current_->data_.count(),  prefetch_current_->data_.gpu_data(),  top[0]->mutable_gpu_data());
  caffe_copy(prefetch_current_->datap_.count(), prefetch_current_->datap_.gpu_data(), top[0]->mutable_gpu_data()+prefetch_current_->data_.count());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    vector<int> shape = prefetch_current_->label_.shape();
    CHECK_LT(shape.size(), 2);
    CHECK_EQ(top[1]->count(), prefetch_current_->label_.count()*2);
    shape[0] *= 2;
    top[1]->Reshape(shape);
    // Copy the labels.
    caffe_copy(prefetch_current_->label_.count(),  prefetch_current_->label_.gpu_data(),  top[1]->mutable_gpu_data());
    caffe_copy(prefetch_current_->labelp_.count(), prefetch_current_->labelp_.gpu_data(), top[1]->mutable_gpu_data()+prefetch_current_->label_.count());
  }
  if (this->output_overlaps_) {
    // Reshape to loaded overlaps.
    vector<int> shape = prefetch_current_->overlap_.shape();
    CHECK_LT(shape.size(), 2);
    CHECK_EQ(top[1]->count(), prefetch_current_->overlap_.count()*2);
    shape[0] *= 2;
    top[2]->Reshape(shape);
    // Copy the overlap.
    caffe_copy(prefetch_current_->overlap_.count(),  prefetch_current_->overlap_.gpu_data(),  top[2]->mutable_gpu_data());
    caffe_copy(prefetch_current_->overlapp_.count(), prefetch_current_->overlapp_.gpu_data(), top[2]->mutable_gpu_data()+prefetch_current_->overlap_.count());
  }
}
INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ReidPrefetchingDataLayer);

}  // namespace caffe
