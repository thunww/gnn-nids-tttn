# notebooks

Notebook Colab (.ipynb), version control qua git (không phải Drive sync). Nguồn thực thi là Colab (GPU miễn phí); logic ổn định nên viết trong `src/` và import vào notebook, không viết thẳng trong cell.

- `00_colab_bootstrap.ipynb` — chạy đầu mỗi phiên: mount Drive (data/checkpoint/mlflow), clone/pull repo từ GitHub, cài `requirements.txt`, thêm `src/` vào `sys.path`.
