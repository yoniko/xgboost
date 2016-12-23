/*!
 * Copyright 2015 by Contributors
 * \file multi_class.cc
 * \brief Definition of multi-class classification objectives.
 * \author Tianqi Chen
 */
#include <dmlc/omp.h>
#include <dmlc/parameter.h>
#include <xgboost/logging.h>
#include <xgboost/objective.h>
#include <vector>
#include <algorithm>
#include <utility>
#include "../common/math.h"
#include "../common/random.h"

namespace xgboost {
namespace obj {

DMLC_REGISTRY_FILE_TAG(multiclass_obj);

struct SoftmaxMultiClassParam : public dmlc::Parameter<SoftmaxMultiClassParam> {
  int num_class;
  // declare parameters
  DMLC_DECLARE_PARAMETER(SoftmaxMultiClassParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
        .describe("Number of output class in the multi-class classification.");
  }
};

class SoftmaxMultiClassObj : public ObjFunction {
 public:
  explicit SoftmaxMultiClassObj(bool output_prob)
      : output_prob_(output_prob) {
  }
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<bst_float>& preds,
                   const MetaInfo& info,
                   int iter,
                   std::vector<bst_gpair>* out_gpair) override {
    CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
    CHECK(preds.size() == (static_cast<size_t>(param_.num_class) * info.labels.size()))
        << "SoftmaxMultiClassObj: label size and pred size does not match";
    out_gpair->resize(preds.size());
    const int nclass = param_.num_class;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nclass);

    int label_error = 0;
    #pragma omp parallel
    {
      std::vector<bst_float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong i = 0; i < ndata; ++i) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[i * nclass + k];
        }
        common::Softmax(&rec);
        int label = static_cast<int>(info.labels[i]);
        if (label < 0 || label >= nclass)  {
          label_error = label; label = 0;
        }
        const bst_float wt = info.GetWeight(i);
        for (int k = 0; k < nclass; ++k) {
          bst_float p = rec[k];
          const bst_float h = 2.0f * p * (1.0f - p) * wt;
          if (label == k) {
            out_gpair->at(i * nclass + k) = bst_gpair((p - 1.0f) * wt, h);
          } else {
            out_gpair->at(i * nclass + k) = bst_gpair(p* wt, h);
          }
        }
      }
    }
    CHECK(label_error >= 0 && label_error < nclass)
        << "SoftmaxMultiClassObj: label must be in [0, num_class),"
        << " num_class=" << nclass
        << " but found " << label_error << " in label.";
  }
  void PredTransform(std::vector<bst_float>* io_preds) override {
    this->Transform(io_preds, output_prob_);
  }
  void EvalTransform(std::vector<bst_float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  const char* DefaultEvalMetric() const override {
    return "merror";
  }

 private:
  inline void Transform(std::vector<bst_float> *io_preds, bool prob) {
    std::vector<bst_float> &preds = *io_preds;
    std::vector<bst_float> tmp;
    const int nclass = param_.num_class;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nclass);
    if (!prob) tmp.resize(ndata);

    #pragma omp parallel
    {
      std::vector<bst_float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong j = 0; j < ndata; ++j) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[j * nclass + k];
        }
        if (!prob) {
          tmp[j] = static_cast<bst_float>(
              common::FindMaxIndex(rec.begin(), rec.end()) - rec.begin());
        } else {
          common::Softmax(&rec);
          for (int k = 0; k < nclass; ++k) {
            preds[j * nclass + k] = rec[k];
          }
        }
      }
    }
    if (!prob) preds = tmp;
  }
  // output probability
  bool output_prob_;
  // parameter
  SoftmaxMultiClassParam param_;
};

// objective for lambda rank
class LambdaRankObj : public ObjFunction {
 public:
  void Configure(const std::vector<std::pair<std::string, std::string> >& args) override {
    param_.InitAllowUnknown(args);
  }
  void GetGradient(const std::vector<float>& preds,
                   const MetaInfo& info,
                   int iter,
                   std::vector<bst_gpair>* out_gpair) override {
    CHECK_NE(info.labels.size(), 0) << "label set cannot be empty";
    CHECK_EQ(preds.size(), info.labels.size()) << "label size predict size not match " << preds.size() << " " << info.labels.size();
    std::vector<bst_gpair>& gpair = *out_gpair;
    gpair.resize(preds.size());
    // quick consistency when group is not available
    /*std::vector<unsigned> tgptr(2, 0); tgptr[1] = static_cast<unsigned>(info.labels.size());
    const std::vector<unsigned> &gptr = info.group_ptr.size() == 0 ? tgptr : info.group_ptr;
    CHECK(gptr.size() != 0 && gptr.back() == info.labels.size())
        << "group structure not consistent with #rows";
    const bst_omp_uint ngroup = static_cast<bst_omp_uint>(gptr.size() - 1);*/
    const int nclass = param_.num_class;
    const bst_omp_uint ngroup = static_cast<omp_ulong>(preds.size() / nclass);
    #pragma omp parallel
    {
      // parall construct, declare random number generator here, so that each
      // thread use its own random number generator, seed by thread id and current iteration
      common::RandomEngine rnd(iter * 1111 + omp_get_thread_num());

      std::vector<LambdaPair> pairs;
      std::vector<ListEntry>  lst;
      std::vector< std::pair<float, unsigned> > rec;
      #pragma omp for schedule(static)
      for (bst_omp_uint k = 0; k < ngroup; ++k) {
        lst.clear(); pairs.clear();
        for (unsigned j = k*nclass; j < (k+1)*nclass; ++j) {
          gpair[j] = bst_gpair(0.0f, 0.0f);
          if(info.labels[j] <= 1.5f)
            lst.push_back(ListEntry(preds[j], info.labels[j], j));
        }
        std::sort(lst.begin(), lst.end(), ListEntry::CmpPred);
        rec.resize(lst.size());
        for (unsigned i = 0; i < lst.size(); ++i) {
          rec[i] = std::make_pair(lst[i].label, i);
        }
        std::sort(rec.begin(), rec.end(), common::CmpFirst);
        // enumerate buckets with same label, for each item in the lst, grab another sample randomly
        /*for (unsigned i = 0; i < rec.size(); ) {
          unsigned j = i + 1;
          while (j < rec.size() && rec[j].first == rec[i].first) ++j;
          // bucket in [i,j), get a sample outside bucket
          unsigned nleft = i, nright = static_cast<unsigned>(rec.size() - j);
          if (nleft + nright != 0) {
            int nsample = 1;//param_.num_pairsample;
            while (nsample --) {
              for (unsigned pid = i; pid < j; ++pid) {
                unsigned ridx = std::uniform_int_distribution<unsigned>(0, nleft + nright - 1)(rnd);
                if (ridx < nleft) {
                  pairs.push_back(LambdaPair(rec[ridx].second, rec[pid].second));
                } else {
                  pairs.push_back(LambdaPair(rec[pid].second, rec[ridx+j-i].second));
                }
              }
            }
          }
          i = j;
        }*/
        unsigned num_positive = 0;
        while((num_positive<rec.size())&(rec[num_positive].first > 0.0f))
          num_positive++;
        for(unsigned pos=0; pos<num_positive; pos++) {
          for(unsigned neg=num_positive; neg<rec.size(); neg++) {
            pairs.push_back(LambdaPair(rec[pos].second, rec[neg].second));
          }
        }

        // get lambda weight for the pairs
        this->GetLambdaWeight(lst, &pairs);
        // rescale each gradient and hessian so that the lst have constant weighted
        float scale = 1.0f / 1;//param_.num_pairsample;
        /*if (param_.fix_list_weight != 0.0f) {
          scale *= param_.fix_list_weight / (gptr[k + 1] - gptr[k]);
        }*/
        for (size_t i = 0; i < pairs.size(); ++i) {
          const ListEntry &pos = lst[pairs[i].pos_index];
          const ListEntry &neg = lst[pairs[i].neg_index];
          const float w = pairs[i].weight * scale;
          const float eps = 1e-16f;
          float p = common::Sigmoid(pos.pred - neg.pred);
          float g = p - 1.0f;
          float h = std::max(p * (1.0f - p), eps);
          // accumulate gradient and hessian in both pid, and nid
          gpair[pos.rindex].grad += g * w;
          gpair[pos.rindex].hess += 2.0f * w * h;
          gpair[neg.rindex].grad -= g * w;
          gpair[neg.rindex].hess += 2.0f * w * h;
        }
      }
    }
  }
  const char* DefaultEvalMetric(void) const override {
    return "map";
  }
  void PredTransform(std::vector<float>* io_preds) override {
    this->Transform(io_preds, true);
  }
  void EvalTransform(std::vector<float>* io_preds) override {
    this->Transform(io_preds, true);
  }

 private:
  inline void Transform(std::vector<float> *io_preds, bool prob) {
    std::vector<float> &preds = *io_preds;
    std::vector<float> tmp;
    const int nclass = param_.num_class;
    const omp_ulong ndata = static_cast<omp_ulong>(preds.size() / nclass);
    if (!prob) tmp.resize(ndata);

    #pragma omp parallel
    {
      std::vector<float> rec(nclass);
      #pragma omp for schedule(static)
      for (omp_ulong j = 0; j < ndata; ++j) {
        for (int k = 0; k < nclass; ++k) {
          rec[k] = preds[j * nclass + k];
        }
        if (!prob) {
          tmp[j] = static_cast<float>(
              common::FindMaxIndex(rec.begin(), rec.end()) - rec.begin());
        } else {
          common::Softmax(&rec);
          for (int k = 0; k < nclass; ++k) {
            preds[j * nclass + k] = rec[k];
          }
        }
      }
    }
    if (!prob) preds = tmp;
  }

 protected:
  /*! \brief helper information in a list */
  struct ListEntry {
    /*! \brief the predict score we in the data */
    float pred;
    /*! \brief the actual label of the entry */
    float label;
    /*! \brief row index in the data matrix */
    unsigned rindex;
    // constructor
    ListEntry(float pred, float label, unsigned rindex)
        : pred(pred), label(label), rindex(rindex) {}
    // comparator by prediction
    inline static bool CmpPred(const ListEntry &a, const ListEntry &b) {
      return a.pred > b.pred;
    }
    // comparator by label
    inline static bool CmpLabel(const ListEntry &a, const ListEntry &b) {
      return a.label > b.label;
    }
  };
  /*! \brief a pair in the lambda rank */
  struct LambdaPair {
    /*! \brief positive index: this is a position in the list */
    unsigned pos_index;
    /*! \brief negative index: this is a position in the list */
    unsigned neg_index;
    /*! \brief weight to be filled in */
    float weight;
    // constructor
    LambdaPair(unsigned pos_index, unsigned neg_index)
        : pos_index(pos_index), neg_index(neg_index), weight(1.0f) {}
  };
  /*!
   * \brief get lambda weight for existing pairs
   * \param list a list that is sorted by pred score
   * \param io_pairs record of pairs, containing the pairs to fill in weights
   */
  virtual void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                               std::vector<LambdaPair> *io_pairs) = 0;

 private:
  SoftmaxMultiClassParam param_;
};

class LambdaRankObjMAP : public LambdaRankObj {
 protected:
  struct MAPStats {
    /*! \brief the accumulated precision */
    float ap_acc;
    /*!
     * \brief the accumulated precision,
     *   assuming a positive instance is missing
     */
    float ap_acc_miss;
    /*!
     * \brief the accumulated precision,
     * assuming that one more positive instance is inserted ahead
     */
    float ap_acc_add;
    /* \brief the accumulated positive instance count */
    float hits;
    MAPStats(void) {}
    MAPStats(float ap_acc, float ap_acc_miss, float ap_acc_add, float hits)
        : ap_acc(ap_acc), ap_acc_miss(ap_acc_miss), ap_acc_add(ap_acc_add), hits(hits) {}
  };
  /*!
   * \brief Obtain the delta MAP if trying to switch the positions of instances in index1 or index2
   *        in sorted triples
   * \param sorted_list the list containing entry information
   * \param index1,index2 the instances switched
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline float GetLambdaMAP(const std::vector<ListEntry> &sorted_list,
                            int index1, int index2,
                            std::vector<MAPStats> *p_map_stats) {
    std::vector<MAPStats> &map_stats = *p_map_stats;
    if (index1 == index2 || map_stats[map_stats.size() - 1].hits == 0) {
      return 0.0f;
    }
    if (index1 > index2) std::swap(index1, index2);
    float original = map_stats[index2].ap_acc;
    if (index1 != 0) original -= map_stats[index1 - 1].ap_acc;
    float changed = 0;
    float label1 = sorted_list[index1].label > 0.0f ? 1.0f : 0.0f;
    float label2 = sorted_list[index2].label > 0.0f ? 1.0f : 0.0f;
    if (label1 == label2) {
      return 0.0;
    } else if (label1 < label2) {
      changed += map_stats[index2 - 1].ap_acc_add - map_stats[index1].ap_acc_add;
      changed += (map_stats[index1].hits + 1.0f) / (index1 + 1);
    } else {
      changed += map_stats[index2 - 1].ap_acc_miss - map_stats[index1].ap_acc_miss;
      changed += map_stats[index2].hits / (index2 + 1);
    }
    float ans = (changed - original) / (map_stats[map_stats.size() - 1].hits);
    if (ans < 0) ans = -ans;
    return ans;
  }
  /*
   * \brief obtain preprocessing results for calculating delta MAP
   * \param sorted_list the list containing entry information
   * \param map_stats a vector containing the accumulated precisions for each position in a list
   */
  inline void GetMAPStats(const std::vector<ListEntry> &sorted_list,
                          std::vector<MAPStats> *p_map_acc) {
    std::vector<MAPStats> &map_acc = *p_map_acc;
    map_acc.resize(sorted_list.size());
    float hit = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    for (size_t i = 1; i <= sorted_list.size(); ++i) {
      if (sorted_list[i - 1].label > 0.0f) {
        hit++;
        acc1 += hit / i;
        acc2 += (hit - 1) / i;
        acc3 += (hit + 1) / i;
      }
      map_acc[i - 1] = MAPStats(acc1, acc2, acc3, hit);
    }
  }
  void GetLambdaWeight(const std::vector<ListEntry> &sorted_list,
                       std::vector<LambdaPair> *io_pairs) override {
    std::vector<LambdaPair> &pairs = *io_pairs;
    std::vector<MAPStats> map_stats;
    GetMAPStats(sorted_list, &map_stats);
    for (size_t i = 0; i < pairs.size(); ++i) {
      pairs[i].weight =
          GetLambdaMAP(sorted_list, pairs[i].pos_index,
                       pairs[i].neg_index, &map_stats);
    }
  }
};

// register the ojective functions
DMLC_REGISTER_PARAMETER(SoftmaxMultiClassParam);

XGBOOST_REGISTER_OBJECTIVE(SoftmaxMultiClass, "multi:softmax")
.describe("Softmax for multi-class classification, output class index.")
.set_body([]() { return new SoftmaxMultiClassObj(false); });

XGBOOST_REGISTER_OBJECTIVE(SoftprobMultiClass, "multi:softprob")
.describe("Softmax for multi-class classification, output probability distribution.")
.set_body([]() { return new SoftmaxMultiClassObj(true); });

XGBOOST_REGISTER_OBJECTIVE(LambdaRankObjMAP, "multi:map")
.describe("LambdaRank with MAP as objective for multiclass.")
.set_body([]() { return new LambdaRankObjMAP(); });

}  // namespace obj
}  // namespace xgboost
