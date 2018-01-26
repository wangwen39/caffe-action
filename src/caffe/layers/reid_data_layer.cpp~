#include <stdint.h>
#include <cfloat>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/reid_data_layer.hpp"
#include <boost/thread.hpp>
#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/image_io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ReidDataLayer<Dtype>::~ReidDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
unsigned int ReidDataLayer<Dtype>::RandRng() {
  CHECK(prefetch_rng_);
  caffe::rng_t *prefetch_rng =
      static_cast<caffe::rng_t *>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

template <typename Dtype>
void ReidDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  DLOG(INFO) << "ReidDataLayer : DataLayerSetUp";
  const int new_length = this->layer_param_.reid_data_param().new_length();
  const int new_height = this->layer_param_.reid_data_param().new_height();
  const int new_width  = this->layer_param_.reid_data_param().new_width();
  string root_folder = this->layer_param_.reid_data_param().root_folder();
  // const bool is_color  = this->layer_param_.reid_data_param().is_color();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the list file
  const string& source = this->layer_param_.reid_data_param().source();
  //const bool use_image = this->layer_param_.reid_data_param().use_image();
  const int sampling_rate = this->layer_param_.reid_data_param().sampling_rate();

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename, labels;
  // float start_frm, label, individual_sampling_rate, overlap;
  int mx_label = -1;
  int mi_label = INT_MAX;

  vector<float> meta(4,0);
  // start_frm >> label >> individual_sampling_rate >> overlap
  while (infile >> filename >> meta[0] >> meta[1] >> meta[2] >> meta[3]) {
    this->lines_.push_back(std::make_pair(filename, meta));
    mx_label = std::max(mx_label, int(meta[1]));
    mi_label = std::min(mi_label, int(meta[1]));
  }
  CHECK_EQ(mi_label, 0);
  this->label_set.clear();
  this->label_set.resize(mx_label+1);
  for (size_t index = 0; index < this->lines_.size(); index++) {
    int label = this->lines_[index].second[1];
    this->label_set[label].push_back(index);
  }
  for (size_t index = 0; index < this->label_set.size(); index++) {
    CHECK_GT(this->label_set[index].size(), 0) << "label : " << index << " has no segments";
  }

  CHECK(!lines_.empty()) << "File is empty";
  infile.close();

  LOG(INFO) << "A total of " << lines_.size() << " segments. Label : [" << mi_label << ", " << mx_label << "]";
  LOG(INFO) << "A total of " << label_set.size() << " persons";
  const unsigned int prefetch_rng_seed = caffe_rng_rand();
  prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));

  this->left_segments = this->lines_.size();
  this->pos_fraction = this->layer_param_.reid_data_param().pos_fraction();
  this->neg_fraction = this->layer_param_.reid_data_param().pos_fraction();

  CHECK_GT(lines_.size(), 0);
  // Read an segment, and use it to initialize the top blob.
  VolumeDatum datum;
  CHECK(ReadImageSequenceToVolumeDatum((root_folder + lines_[0].first).c_str(), int(lines_[0].second[0]), int(lines_[0].second[1]),
                              new_length, new_height, new_width, sampling_rate, &datum, lines_[0].second[3]));   // 030317 no change -> overlap added

  const int batch_size = this->layer_param_.reid_data_param().batch_size();
  // Use data_transformer to infer the expected blob shape from datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  vector<int> prefetch_top_shape = top_shape;
  this->transformed_data_.Reshape(top_shape);
  
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size * 2;
  prefetch_top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  //top[1]->Reshape(top_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(prefetch_top_shape);
    this->prefetch_[i]->datap_.Reshape(prefetch_top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->shape_string();
  //LOG(INFO) << "output data pair size: " << top[1]->num() << ","
  //    << top[1]->channels() << "," << top[1]->height() << ","
  //    << top[1]->width();
  // label
  if (this->output_labels_) {
    vector<int> label_shape(1, batch_size*2);
    top[1]->Reshape(label_shape);
    vector<int> prefetch_label_shape(1, batch_size);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->label_.Reshape(prefetch_label_shape);
      this->prefetch_[i]->labelp_.Reshape(prefetch_label_shape);
    }
    LOG(INFO) << "output label size : " << top[1]->shape_string();
  }

  // overlap 030317
  if (this->output_overlaps_) {
    vector<int> overlap_shape(1, batch_size*2);
    top[2]->Reshape(overlap_shape);
    vector<int> prefetch_overlap_shape(1, batch_size);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
      this->prefetch_[i]->overlap_.Reshape(prefetch_overlap_shape);
      this->prefetch_[i]->overlapp_.Reshape(prefetch_overlap_shape);
    }
    LOG(INFO) << "output label size : " << top[2]->shape_string();
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void ReidDataLayer<Dtype>::load_batch(ReidBatch<Dtype>* batch) {

  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  const int batch_size = this->layer_param_.reid_data_param().batch_size();
  const vector<size_t> batches = this->batch_ids();
  const vector<size_t> batches_pair = this->batch_pairs(batches);
  CHECK_EQ(batches.size(), batch_size);
  CHECK_EQ(batches_pair.size(), batch_size);
  const int new_length = this->layer_param_.reid_data_param().new_length();
  const int new_height = this->layer_param_.reid_data_param().new_height();
  const int new_width  = this->layer_param_.reid_data_param().new_width();
  string root_folder = this->layer_param_.reid_data_param().root_folder();
  const bool show_data = this->layer_param_.reid_data_param().show_data();
  // const bool is_color  = this->layer_param_.reid_data_param().is_color();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  //const bool use_image = this->layer_param_.reid_data_param().use_image();
  const int sampling_rate = this->layer_param_.reid_data_param().sampling_rate();
  // Reshape according to the first segment of each batch
  // on single input batches allows for inputs of varying dimension.
  VolumeDatum datum, datump;
  CHECK(ReadImageSequenceToVolumeDatum((root_folder + lines_[0].first).c_str(), int(lines_[0].second[0]), int(lines_[0].second[1]),
                              new_length, new_height, new_width, sampling_rate, &datum, lines_[0].second[3]));   // 030317 no change -> overlap added

  // Use data_transformer to infer the expected blob shape from a datum.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);
  batch->datap_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_datap = batch->datap_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();
  Dtype* prefetch_labelp = batch->labelp_.mutable_cpu_data();
  Dtype* prefetch_overlap = batch->overlap_.mutable_cpu_data();
  Dtype* prefetch_overlapp = batch->overlapp_.mutable_cpu_data();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    const size_t true_idx = batches[item_id];
    const size_t pair_idx = batches_pair[item_id];

    CHECK(ReadImageSequenceToVolumeDatum((root_folder + lines_[true_idx].first).c_str(), int(lines_[true_idx].second[0]), int(lines_[true_idx].second[1]),
                              new_length, new_height, new_width, sampling_rate*int(lines_[true_idx].second[2]), &datum, lines_[true_idx].second[3]));
    
    CHECK(ReadImageSequenceToVolumeDatum((root_folder + lines_[pair_idx].first).c_str(), int(lines_[pair_idx].second[0]), int(lines_[pair_idx].second[1]),
                              new_length, new_height, new_width, sampling_rate*int(lines_[pair_idx].second[2]), &datump, lines_[pair_idx].second[3]));

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the segment
    vector<int> offset_shape(5,0);
    offset_shape[0] = item_id;
    const int t_offset = batch->data_.offset(offset_shape);
    this->transformed_data_.set_cpu_data(prefetch_data + t_offset);
    this->data_transformer_->VideoTransform(datum, &(this->transformed_data_));

    // Pair Data
    const int p_offset = batch->datap_.offset(offset_shape);
    this->transformed_data_.set_cpu_data(prefetch_datap + p_offset);
    this->data_transformer_->VideoTransform(datump, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();

    // Label
    CHECK_GE(int(lines_[true_idx].second[1]), 0);
    CHECK_GE(int(lines_[pair_idx].second[1]), 0);
    CHECK_LT(int(lines_[true_idx].second[1]), this->label_set.size());
    CHECK_LT(int(lines_[pair_idx].second[1]), this->label_set.size());
    prefetch_label[item_id]    = int(lines_[true_idx].second[1]);
    prefetch_labelp[item_id]   = int(lines_[pair_idx].second[1]);

    // Overlap
    CHECK_GE(lines_[true_idx].second[3], 0);
    CHECK_GE(lines_[pair_idx].second[3], 0);
    CHECK_LE(lines_[true_idx].second[3], 1);
    CHECK_LE(lines_[pair_idx].second[3], 1);
    prefetch_overlap[item_id]    = lines_[true_idx].second[3];
    prefetch_overlapp[item_id]   = lines_[pair_idx].second[3];

    // Show visualization
    if (show_data){
    	const Dtype* data_buffer = (Dtype*)(prefetch_data + t_offset);
        const Dtype* datap_buffer = (Dtype*)(prefetch_datap + p_offset);
        int image_size, channel_size;
       	image_size = top_shape[3] * top_shape[4];
        channel_size = top_shape[2] * image_size;
        for (int l = 0; l < top_shape[2]; ++l) {
        	for (int c = 0; c < 1; ++c) {
        		cv::Mat img;
        		char ch_name[64];
        		BufferToGrayImage(data_buffer + c * channel_size + l * image_size, top_shape[3], top_shape[4], &img);
        		sprintf(ch_name, "Channel %d, data", c);
        		cv::namedWindow(ch_name, CV_WINDOW_AUTOSIZE);
        		cv::imshow(ch_name, img);
                        BufferToGrayImage(datap_buffer + c * channel_size + l * image_size, top_shape[3], top_shape[4], &img);
        		sprintf(ch_name, "Channel %d, data_p", c);
        		cv::namedWindow(ch_name, CV_WINDOW_AUTOSIZE);
        		cv::imshow(ch_name, img);
        	}
        	cv::waitKey(100);
        }
      LOG(INFO) << "Idx : " << item_id << " : (" << prefetch_label[item_id] <<"," << prefetch_overlap[item_id] << ")"\
                                        " vs (" << prefetch_labelp[item_id] <<"," << prefetch_overlapp[item_id] << ")";
    }

  }
  batch_timer.Stop();
  DLOG(INFO) << "Pair Idx : (" << batches[0] << "," << batches_pair[0] << ")";
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
  
}

INSTANTIATE_CLASS(ReidDataLayer);
REGISTER_LAYER_CLASS(ReidData);
//LOG(INFO) << "Idx : " << item_id << " : (" << prefetch_label[item_id] <<"," << prefetch_overlap[item_id] << ")" vs "(" << prefetch_labelp[item_id] <<"," << prefetch_overlapp[item_id] << ")";

}  // namespace caffe
