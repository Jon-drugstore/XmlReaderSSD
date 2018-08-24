#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/my_annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include "caffe/util/im_transforms.hpp"
#include "caffe/common.hpp"

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"


DEFINE_bool(include_background, false,
    "When this option is on, include none_of_the_above as class 0.");
DEFINE_string(delimiter, " ",
    "The delimiter used to separate fields in label_map_file.");


namespace caffe {

template <typename Dtype>
MyAnnotatedDataLayer<Dtype>::~MyAnnotatedDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void MyAnnotatedDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

  lines_id_ = 0;
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
    batch_samplers_.push_back(anno_data_param.batch_sampler(i));
  }
  label_map_file_ = anno_data_param.label_map_file();
  // Make sure dimension is consistent within batch.
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();
  if (transform_param.has_resize_param()) {
    if (transform_param.resize_param().resize_mode() ==
        ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
      CHECK_EQ(batch_size, 1)
        << "Only support batch size of 1 for FIT_SMALL_SIZE.";
    }
  }

  size_t posL;
  size_t posI;

  string xmlDir = anno_data_param.xmldir();
  string imgDir = anno_data_param.imagedir();
  
  boost::filesystem::path path_p(imgDir);
  int a = 0; 

  for(boost::filesystem::directory_iterator it(path_p);
     it != boost::filesystem::directory_iterator{}; ++it){
     
     string tmp = (it->path().filename().string());
     posL = tmp.find_last_of('.');
     string num = tmp.substr(0, posL);

     string xml_file = xmlDir + "/" + num + ".xml";
     string img_file =   imgDir + "/" + tmp;
     lines_.push_back(std::make_pair(img_file, xml_file));
  }

   // bool ReadRichImageToAnnotatedDatum(const string& filename,
   // const string& labelfile, const int height, const int width,
   // const int min_dim, const int max_dim, const bool is_color,
   // const string& encoding, const AnnotatedDatum_AnnotationType type,
   // const string& labeltype, const std::map<string, int>& name_to_label,
   // AnnotatedDatum* anno_datum) {

  // Read a data point, and use it to initialize the top blob.
  int const img_height = transform_param.resize_param().height();
  int const img_width =  transform_param.resize_param().width();
  const string labeltype = "xml";
  const bool is_color = true;

  LabelMap* label_map = new LabelMap();
  MyReadLabelFileToLabelMap(label_map_file_, false, " ", label_map);
  for (int i = 0; i < label_map->item_size(); i++){
      name_to_label_tmp.insert(std::pair<string, int>(label_map->item(i).name(), label_map->item(i).label() ) );
  }

  std::map<string, int>::iterator it;

  //const std::map<string, int> name_to_label = name_to_label_tmp;
  
  string img_file_tmp = lines_[lines_id_].first;
  size_t pos1 = img_file_tmp.find_last_of(".");
  size_t pos2 = img_file_tmp.find_last_of(" ");
  const string encoding = img_file_tmp.substr(pos1, pos2); 

  AnnotatedDatum anno_datum; 
  ReadRichImageToAnnotatedDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                                img_height, img_width, is_color, encoding, AnnotatedDatum_AnnotationType_BBOX,
                                labeltype, name_to_label_tmp, &anno_datum);

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape top[0] and prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  top[0]->Reshape(top_shape);
  for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
    this->prefetch_[i].data_.Reshape(top_shape);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  if (this->output_labels_) {
    has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
    vector<int> label_shape(4, 1);
    //for(int i=100;i;i--)
    //LOG(INFO) << "anno: " << has_anno_type_;
    has_anno_type_ = 1;
    if (has_anno_type_) {
      anno_type_ = anno_datum.type();
      if (anno_data_param.has_anno_type()) {
        // If anno_type is provided in AnnotatedDataParameter, replace
        // the type stored in each individual AnnotatedDatum.
        LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
        anno_type_ = anno_data_param.anno_type();
      }
      // Infer the label shape from anno_datum.AnnotationGroup().
      int num_bboxes = 0;
      if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
        // Since the number of bboxes can be different for each image,
        // we store the bbox information in a specific format. In specific:
        // All bboxes are stored in one spatial plane (num and channels are 1)
        // And each row contains one and only one box in the following format:
        // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
        // Note: Refer to caffe.proto for details about group_label and
        // instance_id.
        for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
          num_bboxes += anno_datum.annotation_group(g).annotation_size();
        }
        label_shape[0] = 1;
        label_shape[1] = 1;
        // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
        // cpu_data and gpu_data for consistent prefetch thread. Thus we make
        // sure there is at least one bbox.
        label_shape[2] = std::max(num_bboxes, 1);
        label_shape[3] = 8;
      } else {
        LOG(FATAL) << "Unknown annotation type.";
      }
    } else {
      label_shape[0] = batch_size;
    }
    top[1]->Reshape(label_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
      this->prefetch_[i].label_.Reshape(label_shape);
    }
  }
}

// This function is called on prefetch thread
template<typename Dtype>
void MyAnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());

  // Reshape according to the first anno_datum of each batch
  // on single input batches allows for inputs of varying dimension.
  const int batch_size = this->layer_param_.data_param().batch_size();
  const AnnotatedDataParameter& anno_data_param =
      this->layer_param_.annotated_data_param();
  const TransformationParameter& transform_param =
    this->layer_param_.transform_param();


  AnnotatedDatum anno_datum; //= *(reader_.full().peek());
  int const img_height = transform_param.resize_param().height();
  int const img_width =  transform_param.resize_param().width();
  const string labeltype = "xml";

  const bool is_color = true;

  string img_file_tmp = lines_[lines_id_].first;
  size_t pos1 = img_file_tmp.find_last_of(".");
  size_t pos2 = img_file_tmp.find_last_of(" ");
  const string encoding = img_file_tmp.substr(pos1, pos2); 

  ReadRichImageToAnnotatedDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                                img_height, img_width, is_color, encoding, AnnotatedDatum_AnnotationType_BBOX,
                                labeltype, name_to_label_tmp, &anno_datum);

  // Use data_transformer to infer the expected blob shape from anno_datum.
  vector<int> top_shape =
      this->data_transformer_->InferBlobShape(anno_datum.datum());
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* top_data = batch->data_.mutable_cpu_data();
  Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
  if (this->output_labels_ && !has_anno_type_) {
    top_label = batch->label_.mutable_cpu_data();
  }

  // Store transformed annotation.
  map<int, vector<AnnotationGroup> > all_anno;
  int num_bboxes = 0;

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    timer.Start();
    // get a anno_datum
    //AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));
    lines_id_++;
    if(lines_id_ >= lines_.size()){
      lines_id_ = 0;
    }

   img_file_tmp = lines_[lines_id_].first;
   pos1 = img_file_tmp.find_last_of(".");
   pos2 = img_file_tmp.find_last_of(" ");
   const string encoding = img_file_tmp.substr(pos1, pos2); 

    AnnotatedDatum anno_datum; 
    ReadRichImageToAnnotatedDatum(lines_[lines_id_].first, lines_[lines_id_].second,
                                img_height, img_width, is_color, encoding, AnnotatedDatum_AnnotationType_BBOX,
                                labeltype, name_to_label_tmp, &anno_datum);
  
      read_time += timer.MicroSeconds();
      timer.Start();
      AnnotatedDatum distort_datum;
      AnnotatedDatum* expand_datum = NULL;
      if (transform_param.has_distort_param()) {
        distort_datum.CopyFrom(anno_datum);
        this->data_transformer_->DistortImage(anno_datum.datum(),
                                              distort_datum.mutable_datum());
        if (transform_param.has_expand_param()) {
          expand_datum = new AnnotatedDatum();
          this->data_transformer_->ExpandImage(distort_datum, expand_datum);
        } else {
          expand_datum = &distort_datum;
        }
      } else {
        if (transform_param.has_expand_param()) {
          expand_datum = new AnnotatedDatum();
          this->data_transformer_->ExpandImage(anno_datum, expand_datum);
        } else {
          expand_datum = &anno_datum;
        }
      }

      AnnotatedDatum* sampled_datum = NULL;
      bool has_sampled = false;
      if (batch_samplers_.size() > 0) {
        // Generate sampled bboxes from expand_datum.
        vector<NormalizedBBox> sampled_bboxes;
        GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
        if (sampled_bboxes.size() > 0) {
          // Randomly pick a sampled bbox and crop the expand_datum.
          int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
          sampled_datum = new AnnotatedDatum();
          this->data_transformer_->CropImage(*expand_datum,
                                             sampled_bboxes[rand_idx],
                                             sampled_datum);
          has_sampled = true;
        } else {
          sampled_datum = expand_datum;
        }
      } else {
        sampled_datum = expand_datum;
      }
      CHECK(sampled_datum != NULL);
      timer.Start();
      vector<int> shape =
          this->data_transformer_->InferBlobShape(sampled_datum->datum());
      if (transform_param.has_resize_param()) {
        if (transform_param.resize_param().resize_mode() ==
            ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
          this->transformed_data_.Reshape(shape);
          batch->data_.Reshape(shape);
          top_data = batch->data_.mutable_cpu_data();
        } else {
          CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                shape.begin() + 1));
        }
      } else {
        CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
              shape.begin() + 1));
      }
      // Apply data transformations (mirror, scale, crop...)
      int offset = batch->data_.offset(item_id);
      this->transformed_data_.set_cpu_data(top_data + offset);
      vector<AnnotationGroup> transformed_anno_vec;
      if (this->output_labels_) {
        if (has_anno_type_) {
          // Make sure all data have same annotation type.
          CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
          if (anno_data_param.has_anno_type()) {
            sampled_datum->set_type(anno_type_);
          } else {
            CHECK_EQ(anno_type_, sampled_datum->type()) <<
                "Different AnnotationType.";
          }
          // Transform datum and annotation_group at the same time
          transformed_anno_vec.clear();
          this->data_transformer_->Transform(*sampled_datum,
                                             &(this->transformed_data_),
                                             &transformed_anno_vec);
          if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
            // Count the number of bboxes.
            for (int g = 0; g < transformed_anno_vec.size(); ++g) {
              num_bboxes += transformed_anno_vec[g].annotation_size();
            }
          } else {
            LOG(FATAL) << "Unknown annotation type.";
          }
          all_anno[item_id] = transformed_anno_vec;
        } else {
          this->data_transformer_->Transform(sampled_datum->datum(),
                                             &(this->transformed_data_));
          // Otherwise, store the label from datum.
          CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
          top_label[item_id] = sampled_datum->datum().label();
        }
      } else {
        this->data_transformer_->Transform(sampled_datum->datum(),
                                           &(this->transformed_data_));
      }
      // clear memory
      if (has_sampled) {
        delete sampled_datum;
      }
      if (transform_param.has_expand_param()) {
        delete expand_datum;
      }
      trans_time += timer.MicroSeconds();

    //reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
  }

  // Store "rich" annotation if needed.
  if (this->output_labels_ && has_anno_type_) {
    vector<int> label_shape(4);
    if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
      label_shape[0] = 1;
      label_shape[1] = 1;
      label_shape[3] = 8;
      if (num_bboxes == 0) {
        // Store all -1 in the label.
        label_shape[2] = 1;
        batch->label_.Reshape(label_shape);
        caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
      } else {
        // Reshape the label and store the annotation.
        label_shape[2] = num_bboxes;
        batch->label_.Reshape(label_shape);
        top_label = batch->label_.mutable_cpu_data();
        int idx = 0;
        for (int item_id = 0; item_id < batch_size; ++item_id) {
          const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
          for (int g = 0; g < anno_vec.size(); ++g) {
            const AnnotationGroup& anno_group = anno_vec[g];
            for (int a = 0; a < anno_group.annotation_size(); ++a) {
              const Annotation& anno = anno_group.annotation(a);
              const NormalizedBBox& bbox = anno.bbox();
              top_label[idx++] = item_id;
              top_label[idx++] = anno_group.group_label();
              top_label[idx++] = anno.instance_id();
              top_label[idx++] = bbox.xmin();
              top_label[idx++] = bbox.ymin();
              top_label[idx++] = bbox.xmax();
              top_label[idx++] = bbox.ymax();
              top_label[idx++] = bbox.difficult();

            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Unknown annotation type.";
    }
  }
  timer.Stop();
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MyAnnotatedDataLayer);
REGISTER_LAYER_CLASS(MyAnnotatedData);

}  // namespace caffe
