import h5py
import random
from torch import as_tensor
import torch


# LIMPIEZA PREVIA DE DATOS
# CENTRALIZACION DE MUESTREOS
# ANALISIS Y DESCRIPCION DE DATOS
"""
Codigo para la limpieza y preprocesamiento de datos en formato HDF5.
Usar para muestrear el dataset. 

Implementado en poet
"""
class Preprocess:
    def __init__(self, input_path, output_path, seed=42):
        self.random = random.Random(seed)
        self.input_path = input_path
        self.output_path = output_path

    def run(self, sample_rate=100, replace=False):
        """
        Ejecuta el preprocesamiento según los parámetros.
        - Si replace: limpia los keypoints
        - Si sample_rate < 100: genera un subset
        - Si sample_rate == 100: copia directo
        """
        if replace:
            print("Corrigiendo +117 kp (--fix)")
            self.make_clean_h5_all_groups(
                self.input_path,
                self.output_path,
                clean_keypoints_fn=self.clean_fn
            )
        elif sample_rate < 100:
            print(f"Muestreo aleatorio al {sample_rate}%")
            self.make_small_h5_ratio(
                self.input_path,
                self.output_path,
                sample_ratio=sample_rate / 100
            )
        else:
            print("Limpiando dataset completo")
            self.make_clean_h5_all_groups(
                self.input_path,
                self.output_path,
                clean_keypoints_fn=None
            )

    def make_small_h5_ratio(self, input_path, output_path, sample_ratio=0.05):
        """
        Crea un .hdf5 reducido tomando una fracción aleatoria de los clips por grupo.
        Procesa todos los grupos presentes (ej. dataset1, dataset2...).
        """
        with h5py.File(input_path, "r") as fin, h5py.File(output_path, "w") as fout:
            group_names = list(fin.keys())
            print(f"Grupos encontrados: {group_names}")

            for group_name in group_names:
                print(f"\nProcesando grupo '{group_name}'...")
                fin_group = fin[group_name]
                fout_group = fout.create_group(group_name)

                fout_k = fout_group.create_group("keypoints")
                fout_e = fout_group.create_group("embeddings")
                fout_l = fout_group.create_group("labels") if "labels" in fin_group else None

                clips = list(fin_group["keypoints"].keys())
                total = len(clips)
                n = max(1, int(total * sample_ratio))

                print(f"  🔍 Seleccionando {n} de {total} clips ({sample_ratio*100:.1f}%)")
                selected = random.sample(clips, n)

                for clip in selected:
                    fout_k.create_dataset(clip, data=fin_group["keypoints"][clip][:], compression="gzip")
                    fout_e.create_dataset(clip, data=fin_group["embeddings"][clip][:])
                    if fout_l:
                        fout_l.create_dataset(clip, data=fin_group["labels"][clip][:])

        print(f"\nSaved in: {output_path}")
    
            
            
        def make_clean_h5_all_groups(input_path, output_path, clean_keypoints_fn=None):
            """
            Crea un nuevo archivo HDF5 copiando todos los grupos y clips del original,
            aplicando una función de limpieza a los keypoints.

            - clean_keypoints_fn: función que recibe un array (T, J, 2) y retorna el keypoint limpio.
            """

            with h5py.File(input_path, "r") as fin:
                with h5py.File(output_path, "w") as fout:
                    group_names = list(fin.keys())
                    print(f"Grupos encontrados: {group_names}")

                    for group_name in group_names: # no merece la pena hacer un tqdm, estan muy desbalanceados
                        print(f"\n Procesando grupo '{group_name}'...")
                        fin_group = fin[group_name]
                        fout_group = fout.create_group(group_name)

                        fout_k = fout_group.create_group("keypoints")
                        fout_e = fout_group.create_group("embeddings")
                        fout_l = fout_group.create_group("labels") if "labels" in fin_group else None

                        clips = list(fin_group["keypoints"].keys())
                        print(f" {len(clips)} clips")

                        for clip in clips:
                            keypoints = fin_group["keypoints"][clip][:]
                            if clean_keypoints_fn:
                                keypoints = clean_keypoints_fn(keypoints)

                            fout_k.create_dataset(clip, data=keypoints, compression="gzip")
                            fout_e.create_dataset(clip, data=fin_group["embeddings"][clip][:])
                            if fout_l:
                                fout_l.create_dataset(clip, data=fin_group["labels"][clip][:])

            print(f"Saved in '{output_path}'")



        def clean_fn(keypoints_np):
            def fix_keypoints(keypoints):
                assert keypoints[0].shape[1] != 117, "Se esta limpiando un dataset ya limpiado"
                T, N, _ = keypoints.shape
                filtered = keypoints.clone()[:, 117:, :] 
                return filtered
            
            if isinstance(keypoints_np, torch.Tensor) and keypoints_np.dtype == torch.float32:
                keypoints = keypoints_np
            else:
                keypoints = as_tensor(keypoints_np, dtype=torch.float32)
            cleaned = fix_keypoints(keypoints)
            return cleaned.numpy()
        
        


