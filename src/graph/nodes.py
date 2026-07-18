import pandas as pd


def build_node_ids(
    df: pd.DataFrame,
    src_ip_col: str,
    src_port_col: str,
    dst_ip_col: str,
    dst_port_col: str,
) -> tuple[dict[str, int], pd.Series, pd.Series]:
    """Gan ID so nguyen lien tuc cho moi dinh (cap IP:port) xuat hien trong df.

    Tra ve:
    - node_id: dict tra "IP:port" -> so nguyen (0..N-1)
    - src_key, dst_key: cot "IP:port" tuong ung tung dong, dung de tra node_id o buoc dung canh
    """
    src_key = df[src_ip_col].astype(str) + ":" + df[src_port_col].astype(str)
    dst_key = df[dst_ip_col].astype(str) + ":" + df[dst_port_col].astype(str)

    unique_keys = pd.concat([src_key, dst_key], ignore_index=True).unique()
    node_id = {key: i for i, key in enumerate(unique_keys)}

    return node_id, src_key, dst_key
