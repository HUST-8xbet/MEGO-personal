# Kiểm kê routing MEGO và pipeline TSP/CVRP

Ngày kiểm kê: 2026-10-25. Commit hiện tại: `63785c662dd5a2d8ea91ec8cc1e7552d1f4ce01d`.

Phạm vi chỉ đọc code, checkpoint metadata và log đã có. Kiểm kê này không gọi evaluator/objective mới, không train, không sinh nghiệm và không thay đổi thuật toán.

## Kết luận ngắn

Chưa đủ dữ liệu để kiểm tra giả thuyết “rho trước adaptation dự đoán hiệu quả expert sau adaptation”. Hiện chỉ có một source expert TSP được dùng cho TSP→CVRP; các seed/checkpoint và các nhánh `source`, `shuffle`, `rank_only` là các lần adaptation/đối chứng của cùng nguồn, không phải các source expert độc lập.

Ngoài ra, `rho1/rho2` không tồn tại dưới tên đó trong code hiện tại. Cặp tương ứng của MEGO gốc là Pearson và Spearman trong `DecoderMapping.evaluate_correlation`. Các log hiện có không lưu chúng; cache `all_data_corr*.pkl` cũng không có trong checkout. Vì vậy không thể lập bảng rho trước adaptation mà không thực hiện đánh giá mới.

## 1. MEGO gốc và extension

### Cơ chế gốc

| Thành phần | Bằng chứng code | Cơ chế xác nhận |
|---|---|---|
| Correlation | [`src/decoder_mapping.py:343-369`](../src/decoder_mapping.py:343) | Lấy `target_x,target_y`; đưa `target_x` qua source model; `predict_y` là score head; trả `(Pearson, Spearman)`. |
| Nghiệm vào encoder | [`src/decoder_mapping.py:346-364`](../src/decoder_mapping.py:346) | Nghiệm target được cắt về `source_dimension` hoặc padding 0/1/ngẫu nhiên để có cùng chiều binary với source model. |
| Giá trị thật/dự đoán | [`src/decoder_mapping.py:346-368`](../src/decoder_mapping.py:346) | Giá trị thật là `target_y`; dự đoán là score head của source model trên vector target đã cắt/pad. Spearman được tính trên thứ hạng của hai vector. |
| Chọn expert | [`src/experiment_decoder_mapping.py:77-89`](../src/experiment_decoder_mapping.py:77) và [`:136-141`](../src/experiment_decoder_mapping.py:136) | Chạy từng source instance; lưu Pearson/Spearman; coi expert “positive” khi cả hai > 0. Sau đó so hit của các expert được chọn với random. |
| Ghép sau chọn | [`src/decoder_mapping.py:101-128`](../src/decoder_mapping.py:101) | Sort source/target quality giảm dần, ghép các quality levels tương ứng, rồi shuffle cặp. |
| Budget | [`src/decoder_mapping.py:130-173`](../src/decoder_mapping.py:130) | `source_coefficient * sample_num` source và `sample_num` target; mapping decoder được fine-tune theo config. |
| Lấy nghiệm sau mapping | [`src/decoder_mapping.py:230-341`](../src/decoder_mapping.py:230) | Lấy binary source ngẫu nhiên, chạy mapping model, chọn output theo predicted performance; không phải chọn theo rho tại bước này. |

“rho1/rho2” vì vậy chỉ là cách gọi quy ước cho Pearson/Spearman; code không định nghĩa hai biến có tên rho1/rho2.

### Extension TSP/CVRP

- TSP→CVRP dùng `load_source` trong [`src/experiment_cvrp_mapping.py:23-48`](../src/experiment_cvrp_mapping.py:23): source checkpoint TSP được encode trên tour TSP, lấy posterior mean, rồi đóng băng.
- Pairing extension dùng quality giảm dần theo từng split trong [`src/cvrp_mapping.py:35-66`](../src/cvrp_mapping.py:35), không tính Pearson/Spearman trước adaptation.
- Decoder CVRP đích mới và bridge 16→16 được tạo trong [`src/experiment_cvrp_mapping.py:103-115`](../src/experiment_cvrp_mapping.py:103); mapping loss là reconstruction NLL trên sequence CVRP trong [`src/cvrp_mapping.py:69-113`](../src/cvrp_mapping.py:69).
- Các nhánh `source`, `shuffled_train`, `noise_train`, `rank_only`, `no_source` là các điều kiện của cùng một target adaptation, không phải các source expert khác nhau.
- Within-rank extension kế thừa source TSP và chỉ thay điều kiện latent bằng shuffle trong bin/rank-only; [`src/experiment_cvrp_within_rank.py:180-224`](../src/experiment_cvrp_within_rank.py:180) xác nhận cùng template, cùng split và cùng lịch train.

### Khoảng trống biểu diễn

Không có đường tính rho hợp lệ cho TSP→CVRP trong extension hiện tại. TSP encoder nhận tọa độ và permutation tour (`TSPVAE.encode`), trong khi CVRP biểu diễn là tọa độ + demand/capacity + sequence depot-padded (`CVRPVAE.encode`). Việc cả hai có `dimension=20` không làm chúng cùng biểu diễn: TSP có 20 city tokens; CVRP có 21 vertex tokens và sequence rộng 41.

Code hiện tại không cắt/pad/flatten CVRP vào TSP encoder để tính rho. Thêm thao tác đó sẽ là một representation adapter/encoder mới, ngoài phạm vi nhiệm vụ. Do đó đây là khoảng trống triển khai, chưa phải bằng chứng routing gốc thất bại.

## 2. Expert/checkpoint hiện có

| Expert/checkpoint | Family / instance | Biểu diễn và kích thước | Latent | Checksum chính | Mapping hiện tại |
|---|---|---|---:|---|---|
| `logs_smoke_20261020_224736/MaxCutProblem-0-SurrogateVAE/best_model.pt` | Max-Cut instance `0`, `in_dim=30` | binary vector 30 | 120 | checkpoint legacy không có `data_sha256` trong payload | Nguồn legacy Max-Cut→TSP; không phải expert TSP/CVRP khác |
| `logs_tsp/tsp20_expert_smoke_01/best_model.pt` | TSP20, data instance seed1088 | permutation tour 20; train/valid 1638/410 | 16 | data `44b912...a3a4bd`; checkpoint `94ec28...e719` | **Source duy nhất** trong TSP→CVRP |
| `logs_cvrp/cvrp20_expert_smoke_01/best_model.pt` | CVRP20, data instance seed1088, capacity30 | depot-separated route, padded width41; train/valid 1638/410 | 16 | data `c41108...ac36c`; checkpoint `2c19fc...e4358` | Expert CVRP riêng; weights **không dùng** trong TSP→CVRP |

Split file của hai expert có cùng checksum `c17d29...07d2` vì cùng chỉ số 80/20 seed1088; điều này không biến hai expert thành cùng một bài toán.

Các file sau không được đếm là expert độc lập:

- `logs_tsp/mapping_controls_01/*.pt`: năm arm của một mapping target TSP.
- `logs_cvrp/mapping_tsp_to_cvrp_01/*.pt`: năm arm của một mapping TSP→CVRP.
- `logs_cvrp/within_rank_shuffle_v1/seed_{1088,1089,1090}/*/mapping_checkpoint.pt`: ba training seed × ba arm trên cùng source TSP và target CVRP.
- `tsp_cvrp_coupling_*/*/*.pt`: các arm coupling và training seed, không phải source expert mới.

Khác epoch hoặc khác training seed chỉ là checkpoint/replicate của cùng source instance theo yêu cầu kiểm kê, không phải nguồn tri thức độc lập.

## 3. Dữ liệu routing và kết quả có thể so sánh

### rho trước adaptation

Không có giá trị rho lưu trong:

- `logs_tsp/mapping_controls_01/diagnostics.json`;
- `logs_cvrp/mapping_tsp_to_cvrp_01/diagnostics.json`;
- `logs_cvrp/within_rank_shuffle_v1/diagnostics.json`;
- các `conditioning.json`/checkpoint metadata hiện có.

Các file này lưu pairing, condition, NLL, generated quality và cost, không lưu Pearson/Spearman trước adaptation. `all_data_corr*.pkl` của runner gốc cũng không tồn tại. Vì vậy bảng rho theo expert hiện là **chưa xác định**, không được thay bằng số Pearson quality của expert sau khi train.

### Adaptation cùng target hiện có

| Run | Source độc lập | Target | Protocol/budget | Kết quả có thể dùng |
|---|---|---|---|---|
| `logs_cvrp/mapping_tsp_to_cvrp_01` | 1 TSP20 seed1088 | CVRP20 seed2088 | 2048 cặp; 1638/410; 30 epoch; 5 arm; mỗi arm 410 greedy + 410 sampled | So arm trong cùng target, **không** so expert selection |
| `logs_cvrp/within_rank_shuffle_v1` | cùng 1 TSP20 seed1088 | cùng CVRP20 seed2088 | 3 training seed × 3 arm; 10×41 sampling; 820 target calls/arm/seed; adaptation ~40–42s/arm | So source vs shuffle/rank-only dưới cùng protocol; không có source expert thứ hai |
| `logs_cvrp/tsp_cvrp_coupling_fixed_full_v2` | cùng 1 TSP source | cùng CVRP target | 3 training seed; rank/quality/relation coupling | So coupling, không phải routing giữa experts |

Incumbent adaptation dùng chung trong within-rank là `9.399715696516127`, với `2048` label đã có; source/target verification được ghi riêng, không gọi lại trong kiểm kê.

Các seed quan sát được: adaptation seed `1088,1089,1090`; mapping ban đầu config seed `1088`; target data instance seed `2088`, solution seed `2089`; within-rank generation seed base `95001` và source selection seed `94001`. Các protocol khác nhau không được gộp thành một benchmark.

Một vài kết quả cùng-target chỉ để mô tả phạm vi, không phải routing evidence:

| Run/arm | Sample mean CVRP length | Best sampled |
|---|---:|---:|
| mapping TSP→CVRP `source` | 13.2240 | 10.2371 |
| mapping TSP→CVRP `rank_only` | 13.1823 | 10.0264 |
| mapping TSP→CVRP `no_source` | 13.2077 | 9.8190 |
| within-rank `source`, seed1088/1089/1090 | 13.1955 / 12.9551 / 13.1307 | 9.9834 / 9.3290 / 9.4355 |

Những con số này chỉ so điều kiện trong cùng một target adaptation; không cung cấp biến độc lập “expert identity”.

## 4. Kết luận khả thi

### A–C

- **A: Chưa đủ** expert và kết quả để đánh giá routing. Hiện có một TSP source expert cho CVRP target và một CVRP expert riêng không được dùng trong mapping đó.
- **B: Nếu chỉ xét TSP→CVRP, đúng là chỉ có một source expert.** Ba training seed và các arm đối chứng không thay thế cho nhiều source instance.
- **C: rho gốc không áp dụng trực tiếp cho TSP→CVRP.** Binary vector truncation/padding của MEGO gốc không có ngữ nghĩa hợp lệ cho permutation/route sequence. Đây là thiếu triển khai representation-level, chưa phải kết luận routing không hiệu quả.

### D. Phần thiếu và pilot tối thiểu đề xuất

Chưa chạy pilot. Để phân biệt routing thật với khác biệt seed/protocol, tối thiểu cần:

1. Ít nhất **3 source expert độc lập theo instance** trong cùng family/representation; tốt hơn là 5. Không dùng epoch khác hoặc seed khác để thay thế số instance.
2. Một target cố định có split và incumbent cố định; với mỗi source expert ghi rho trước adaptation trên cùng protocol, hoặc dùng một representation adapter đã định nghĩa trước nếu nghiên cứu cross-family.
3. Mỗi expert chạy cùng adaptation budget, optimizer, decoder initialization policy, generation budget và ít nhất 3 adaptation seeds; có target-only/rank-only control chung.
4. Lưu source instance ID, checkpoint/data/split checksum, rho1/rho2 raw, target adaptation objective, wall time, generation seed và incumbent provenance.
5. Đánh giá trên nhiều target instance hoặc một final-test partition không dùng checkpoint selection. Một target đơn chỉ là pilot cơ chế.

Quy mô tối thiểu có thể phân biệt sơ bộ: `3 experts × 3 adaptation seeds` trên cùng target, cộng target-only và rank-only controls. Đây là ngưỡng pilot, chưa đủ cho tuyên bố tổng quát.

### E. Tiêu chí tiếp tục/dừng

Tiếp tục routing nếu rho trước adaptation có quan hệ định trước, lặp lại trên các expert/seed/target, với cải thiện ngoài rank-only/target-only và bootstrap/interval không chỉ do một target hoặc greedy mode. Cần báo riêng greedy, sampled và best-of-budget.

Dừng hoặc không ưu tiên routing nếu rho không ổn định giữa seed/target, không dự đoán được adaptation outcome ngoài quality rank, hoặc lợi ích chỉ xuất hiện ở một protocol/greedy mode và mất khi giữ cùng ngân sách. Khi đó kết quả ủng hộ quay lại kiểm tra latent selection/representation hoặc dừng ứng viên, không mặc định chuyển hướng kiến trúc.

Không có kết luận novelty hay transfer benefit từ bản kiểm kê này.

## Kiểm tra thực hiện

- Commit đọc: `63785c662dd5a2d8ea91ec8cc1e7552d1f4ce01d`.
- Objective calls mới: **0**.
- File thay đổi: chỉ báo cáo này; không sửa source code, checkpoint, data hay log cũ.
