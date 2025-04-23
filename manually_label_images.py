# import streamlit as st
# import pandas as pd
# import cv2
# from utils.config import Configuration, load_config
# import os
# import streamlit.components.v1 as components

# try:
#     from streamlit_javascript import st_javascript
# except ImportError:
#     st.error(
#         "Please install streamlit_javascript via pip: pip install streamlit_javascript"
#     )
#     st.stop()


# def save_labels(df, target_path):
#     """Save current labels to the Excel file."""
#     df_copy = df.copy()
#     df_copy["label"] = st.session_state.labels

#     df_filtered = df_copy[df_copy["label"] == "Accepted"]

#     base_path = os.path.dirname(target_path)
#     base, ext = os.path.splitext(os.path.basename(target_path))
#     filtered_target_path = os.path.join(base_path, f"{base}_filtered{ext}")

#     df_copy.to_excel(target_path, index=False)
#     df_filtered.to_excel(filtered_target_path, index=False)

#     st.success(f"Labeled data saved to {target_path}")


# def label_images(config: Configuration, df_to_check_path: str, target_path: str):
#     # Read the Excel file with the full image metadata.
#     df = pd.read_excel(df_to_check_path)

#     # --- Initialize session state ---
#     if os.path.exists(target_path):
#         # If a labeled file exists, load its labels if they match the master file.
#         labelled_df = pd.read_excel(target_path)
#         if "label" in labelled_df.columns and len(labelled_df) == len(df):
#             if "labels" not in st.session_state:
#                 st.session_state.labels = labelled_df["label"].tolist()
#             if "current_index" not in st.session_state:
#                 # Set current_index to the first unlabelled sample.
#                 for i, lab in enumerate(st.session_state.labels):
#                     if pd.isna(lab) or lab is None:
#                         st.session_state.current_index = i
#                         break
#                 else:
#                     st.session_state.current_index = len(df)
#         else:
#             if "labels" not in st.session_state:
#                 st.session_state.labels = [None] * len(df)
#             if "current_index" not in st.session_state:
#                 st.session_state.current_index = 0
#     else:
#         if "labels" not in st.session_state:
#             st.session_state.labels = [None] * len(df)
#         if "current_index" not in st.session_state:
#             st.session_state.current_index = 0

#     # If all images have been processed, show summary and save option.
#     if st.session_state.current_index >= len(df):
#         st.write("All images processed!")
#         df["label"] = st.session_state.labels
#         df_filtered = df[df["label"] == "Accepted"]

#         base_path = os.path.dirname(target_path)
#         base, ext = os.path.splitext(os.path.basename(target_path))
#         filtered_target_path = os.path.join(base_path, f"{base}_filtered{ext}")

#         df_filtered.to_excel(filtered_target_path, index=False)

#         st.write(df)
#         if st.button("Save Labeled Data"):
#             df.to_excel(target_path, index=False)
#             st.success(f"Labeled data saved to {target_path}")
#         return

#     st.title("Image Labeler")
#     st.write(f"Image {st.session_state.current_index + 1} of {len(df)}")

#     # --- Capture key events using st_javascript ---
#     # This returns the keyCode when a key is pressed.
#     key_code = st_javascript(
#         """
#         new Promise(resolve => {
#             document.addEventListener('keydown', event => {
#                 // Prevent default behavior (like backspace navigation).
#                 event.preventDefault();
#                 resolve(event.keyCode);
#             }, {once: true});
#         });
#         """
#     )

#     if key_code is not None:
#         if key_code == 32:  # Spacebar pressed.
#             st.session_state.labels[st.session_state.current_index] = "Accepted"
#             save_labels(df, target_path)
#             if st.session_state.current_index < len(df) - 1:
#                 st.session_state.current_index += 1
#             st.rerun()
#         elif key_code == 8:  # Backspace pressed.
#             st.session_state.labels[st.session_state.current_index] = "Skipped"
#             save_labels(df, target_path)
#             if st.session_state.current_index < len(df) - 1:
#                 st.session_state.current_index += 1
#             st.rerun()

#     # --- Manual navigation buttons ---
#     if st.button("Previous"):
#         if st.session_state.current_index > 0:
#             st.session_state.current_index -= 1
#             st.rerun()

#     # --- Show current label status with color ---
#     current_label = st.session_state.labels[st.session_state.current_index]
#     if current_label is None or pd.isna(current_label):
#         st.markdown(
#             "<span style='color: blue;'>Unlabelled</span>", unsafe_allow_html=True
#         )
#     elif current_label == "Accepted":
#         st.markdown(
#             "<span style='color: green;'>Accepted</span>", unsafe_allow_html=True
#         )
#     elif current_label == "Skipped":
#         st.markdown("<span style='color: red;'>Skipped</span>", unsafe_allow_html=True)
#     else:
#         st.write("Unknown label")

#     # --- Load and display the current image ---
#     row = df.iloc[st.session_state.current_index]
#     image_filename = os.path.basename(row["file_name"])
#     img_path = os.path.join(config.directories.images_target_dir, image_filename)
#     image = cv2.imread(img_path)
#     if image is None:
#         st.write(f"Could not read image: {img_path}")
#         st.session_state.labels[st.session_state.current_index] = "Skipped"
#         save_labels(df, target_path)
#         if st.session_state.current_index < len(df) - 1:
#             st.session_state.current_index += 1
#         st.rerun()

#     height, width = image.shape[:2]
#     max_display_size = 800
#     if max(height, width) > max_display_size:
#         scaling = max_display_size / max(height, width)
#         image = cv2.resize(
#             image, None, fx=scaling, fy=scaling, interpolation=cv2.INTER_AREA
#         )

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     st.image(image, caption=image_filename)
#     st.write("Press **Spacebar** to Accept, **Backspace** to Skip.")

#     # --- Fallback manual buttons ---
#     col_accept, col_skip = st.columns(2)
#     with col_accept:
#         if st.button("Accept"):
#             st.session_state.labels[st.session_state.current_index] = "Accepted"
#             save_labels(df, target_path)
#             if st.session_state.current_index < len(df) - 1:
#                 st.session_state.current_index += 1
#             st.rerun()
#     with col_skip:
#         if st.button("Skip"):
#             st.session_state.labels[st.session_state.current_index] = "Skipped"
#             save_labels(df, target_path)
#             if st.session_state.current_index < len(df) - 1:
#                 st.session_state.current_index += 1
#             st.rerun()


# if __name__ == "__main__":
#     faces_set = False

#     # Adjust these paths and configuration as needed.
#     subdir = "subj_08"
#     st.title("Image Labeler")
#     config = load_config(
#         "config.yaml"
#     )  # Ensure your load_config() function is defined and accessible.

#     if faces_set:
#         df_to_check_path = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_face_nans_removed,
#         )
#         target_path = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_face_labelled,
#         )
#     else:
#         df_to_check_path = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_non_face_nans_removed,
#         )
#         target_path = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_non_face_labelled,
#         )

#     label_images(config, df_to_check_path, target_path)

# ------------------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import cv2
# import zipfile
# from utils.config import Configuration, load_config
# import os
# import streamlit.components.v1 as components

# try:
#     from streamlit_javascript import st_javascript
# except ImportError:
#     st.error(
#         "Please install streamlit_javascript via pip: pip install streamlit_javascript"
#     )
#     st.stop()


# def save_labels(df, target_path):
#     """Save current labels to the Excel file."""
#     df_copy = df.copy()
#     df_copy["label"] = st.session_state.labels

#     df_filtered = df_copy[df_copy["label"] == "Accepted"]
#     base_path = os.path.dirname(target_path)
#     base, ext = os.path.splitext(os.path.basename(target_path))
#     filtered_target_path = os.path.join(base_path, f"{base}_filtered{ext}")

#     df_copy.to_excel(target_path, index=False)
#     df_filtered.to_excel(filtered_target_path, index=False)
#     st.success(f"Labeled data saved to {target_path}")


# def label_images(
#     config: Configuration, df_to_check_path: str, target_path: str, face_set: bool
# ):
#     # Load master dataframe
#     df = pd.read_excel(df_to_check_path)

#     # Initialize session state
#     if "face_set" not in st.session_state:
#         st.session_state.face_set = face_set

#     labels_loaded = False
#     if os.path.exists(target_path):
#         try:
#             labelled_df = pd.read_excel(target_path)
#             labels_loaded = True
#         except (FileNotFoundError, zipfile.BadZipFile, ValueError):
#             st.warning(
#                 "Keine bestehende Label-Datei gefunden oder Datei ist beschädigt. Initialisiere neues Labeling..."
#             )

#     if labels_loaded and "label" in labelled_df.columns and len(labelled_df) == len(df):
#         if "labels" not in st.session_state:
#             st.session_state.labels = labelled_df["label"].tolist()
#         if "current_index" not in st.session_state:
#             # Find first unlabelled index
#             for i, lab in enumerate(st.session_state.labels):
#                 if pd.isna(lab) or lab is None:
#                     st.session_state.current_index = i
#                     break
#             else:
#                 st.session_state.current_index = len(df)
#     else:
#         if "labels" not in st.session_state:
#             st.session_state.labels = [None] * len(df)
#         if "current_index" not in st.session_state:
#             st.session_state.current_index = 0

#     # If done
#     if st.session_state.current_index >= len(df):
#         st.write("**All images processed!**")
#         df["label"] = st.session_state.labels
#         st.write(df)
#         if st.button("Save Labeled Data"):
#             df.to_excel(target_path, index=False)
#             st.success(f"Labeled data saved to {target_path}")
#         return

#     # UI
#     st.title("Image Labeler")
#     st.write(f"Image {st.session_state.current_index + 1} of {len(df)}")

#     # Statistics
#     labels = st.session_state.labels
#     if face_set:
#         accepted_count = sum(1 for lab in labels if lab == "Accepted")
#         st.markdown(f"**Faces labeled:** {accepted_count}")
#     else:
#         animals_count = sum(1 for lab in labels if lab == "animals")
#         personen_count = sum(1 for lab in labels if lab == "personen")
#         st.markdown(
#             f"**Animals:** {animals_count}   |   **Personen:** {personen_count}"
#         )

#     # Capture key events via event.code
#     key_code = st_javascript(
#         """
#         new Promise(resolve => {
#             document.addEventListener('keydown', event => {
#                 event.preventDefault();
#                 resolve(event.code);
#             }, {once: true});
#         });
#         """
#     )

#     # Handle only expected codes
#     if isinstance(key_code, str):
#         idx = st.session_state.current_index
#         if st.session_state.face_set:
#             if key_code == "Space":
#                 st.session_state.labels[idx] = "Accepted"
#             elif key_code == "Backspace":
#                 st.session_state.labels[idx] = "Skipped"
#         else:
#             if key_code == "KeyA":
#                 st.session_state.labels[idx] = "animals"
#             elif key_code == "KeyP":
#                 st.session_state.labels[idx] = "personen"
#             elif key_code == "Backspace":
#                 st.session_state.labels[idx] = "Skipped"
#         # If we assigned a label, move on
#         if st.session_state.labels[idx] is not None:
#             save_labels(df, target_path)
#             if idx < len(df) - 1:
#                 st.session_state.current_index += 1
#             st.rerun()

#     # Manual buttons
#     col_back, col1, col2, col_skip = st.columns(4)
#     with col_back:
#         if st.button("Previous") and st.session_state.current_index > 0:
#             st.session_state.current_index -= 1
#             st.rerun()
#     if st.session_state.face_set:
#         with col1:
#             if st.button("Accept (Space)"):
#                 idx = st.session_state.current_index
#                 st.session_state.labels[idx] = "Accepted"
#                 save_labels(df, target_path)
#                 if idx < len(df) - 1:
#                     st.session_state.current_index += 1
#                 st.rerun()
#     else:
#         with col1:
#             if st.button("Animals (A)"):
#                 idx = st.session_state.current_index
#                 st.session_state.labels[idx] = "animals"
#                 save_labels(df, target_path)
#                 if idx < len(df) - 1:
#                     st.session_state.current_index += 1
#                 st.rerun()
#         with col2:
#             if st.button("Personen (P)"):
#                 idx = st.session_state.current_index
#                 st.session_state.labels[idx] = "personen"
#                 save_labels(df, target_path)
#                 if idx < len(df) - 1:
#                     st.session_state.current_index += 1
#                 st.rerun()
#     with col_skip:
#         if st.button("Skip (Backspace)"):
#             idx = st.session_state.current_index
#             st.session_state.labels[idx] = "Skipped"
#             save_labels(df, target_path)
#             if idx < len(df) - 1:
#                 st.session_state.current_index += 1
#             st.rerun()

#     # Display current label
#     current_label = st.session_state.labels[st.session_state.current_index]
#     if current_label is None or pd.isna(current_label):
#         st.markdown(
#             "<span style='color: blue;'>Unlabelled</span>", unsafe_allow_html=True
#         )
#     else:
#         color_map = {
#             "Accepted": "green",
#             "Skipped": "red",
#             "animals": "orange",
#             "personen": "purple",
#         }
#         clr = color_map.get(current_label, "black")
#         st.markdown(
#             f"<span style='color: {clr};'>{current_label}</span>",
#             unsafe_allow_html=True,
#         )

#     # Show image
#     row = df.iloc[st.session_state.current_index]
#     img_name = os.path.basename(row["file_name"])
#     img_path = os.path.join(config.directories.images_target_dir, img_name)
#     image = cv2.imread(img_path)
#     if image is None:
#         st.write(f"Could not read image: {img_path}")
#         st.session_state.labels[st.session_state.current_index] = "Skipped"
#         save_labels(df, target_path)
#         if st.session_state.current_index < len(df) - 1:
#             st.session_state.current_index += 1
#         st.rerun()

#     h, w = image.shape[:2]
#     max_dim = 800
#     if max(h, w) > max_dim:
#         factor = max_dim / max(h, w)
#         image = cv2.resize(
#             image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA
#         )

#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     st.image(image, caption=img_name)
#     if st.session_state.face_set:
#         st.write("Press **Spacebar** to Accept, **Backspace** to Skip.")
#     else:
#         st.write("Press **A** for Animals, **P** for Personen, **Backspace** to Skip.")


# if __name__ == "__main__":
#     faces_set = False  # Toggle here

#     subdir = "subj_01"
#     config = load_config("config.yaml")

#     if faces_set:
#         df_to_check = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_face_nans_removed,
#         )
#         target = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_face_labelled,
#         )
#     else:
#         df_to_check = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_non_face_nans_removed,
#         )
#         target = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_non_face_labelled,
#         )

#     label_images(config, df_to_check, target, faces_set)

# --------------------------------------------
# import streamlit as st
# import pandas as pd
# import cv2
# import os
# from utils.config import Configuration, load_config


# def save_full_and_filtered(df_full, target_full, filtered_full):
#     df_full.to_excel(target_full, index=False)
#     df_full[df_full["label"] == "Accepted"].to_excel(filtered_full, index=False)
#     st.success(f"Saved:\n • {target_full}\n • {filtered_full}")


# def label_images(
#     config: Configuration, df_to_check_path: str, target_path: str, face_set: bool
# ):
#     # 1) DataFrames & Pfade
#     df_full = pd.read_excel(df_to_check_path)
#     base, ext = os.path.splitext(target_path)
#     filtered_path = f"{base}_filtered{ext}"

#     # 2) Session‑State initialisieren
#     if "mode" not in st.session_state:
#         st.session_state.mode = "relabel" if os.path.exists(filtered_path) else "normal"
#         st.session_state.relabel_done = False

#     # 3) Relabel‑Modus
#     if st.session_state.mode == "relabel" and not st.session_state.relabel_done:
#         df_filt = pd.read_excel(filtered_path)
#         if "relabel" not in df_filt.columns:
#             df_filt["relabel"] = None
#         if "to_relabel" not in st.session_state:
#             st.session_state.to_relabel = df_filt[
#                 df_filt["relabel"].isna()
#             ].index.tolist()
#             st.session_state.r_idx = 0

#         total = len(st.session_state.to_relabel)
#         if st.session_state.r_idx >= total:
#             # Merge nachlabel ins Full-DF
#             mapping = df_filt.set_index("file_name")["relabel"].dropna()
#             df_full = df_full.merge(
#                 mapping.rename("newlab"), on="file_name", how="left"
#             )
#             df_full.loc[df_full["newlab"].notna(), "label"] = df_full["newlab"]
#             df_full.drop(columns="newlab", inplace=True)

#             # Speichern unter neuen Namen
#             save_full_and_filtered(
#                 df_full,
#                 target_path.replace(ext, "_postrelabel" + ext),
#                 filtered_path.replace(ext, "_postrelabel_filtered" + ext),
#             )

#             # Wechsel in Normalmodus
#             st.session_state.relabel_done = True
#             st.session_state.mode = "normal"
#             # Labels neu initialisieren
#             st.session_state.pop("labels", None)
#             return  # nächster Durchlauf wird Normalmodus

#         # Einzelbild im Relabel‑Modus
#         idx = st.session_state.to_relabel[st.session_state.r_idx]
#         row = df_filt.loc[idx]
#         _show_and_choose(
#             row=row,
#             config=config,
#             df=df_filt,
#             index=idx,
#             mode="relabel",
#             face_set=None,
#             df_full=None,
#             df_to_check_path=None,
#             target_path=None,
#             filtered_path=filtered_path,
#         )
#         return

#     # 4) Normalmodus (ggf. nach Relabel)
#     if "labels" not in st.session_state:
#         # Alte Labels laden, falls vorhanden
#         if os.path.exists(target_path):
#             prev = pd.read_excel(target_path)
#             if len(prev) == len(df_full) and "label" in prev.columns:
#                 st.session_state.labels = prev["label"].tolist()
#             else:
#                 st.session_state.labels = [None] * len(df_full)
#         else:
#             st.session_state.labels = [None] * len(df_full)
#         # current_index auf erstes None
#         st.session_state.current_index = next(
#             (i for i, lab in enumerate(st.session_state.labels) if pd.isna(lab)),
#             len(df_full),
#         )

#     # Abbruch, wenn alles fertig
#     if st.session_state.current_index >= len(df_full):
#         st.write("✅ Alle Bilder im Normalmodus fertig!")
#         df_full["label"] = st.session_state.labels
#         save_full_and_filtered(df_full, target_path, filtered_path)
#         return

#     # Einzelbild im Normalmodus
#     idx = st.session_state.current_index
#     row = df_full.iloc[idx]
#     _show_and_choose(
#         row=row,
#         config=config,
#         df=df_full,
#         index=idx,
#         mode="normal",
#         face_set=face_set,
#         df_full=df_full,
#         df_to_check_path=df_to_check_path,
#         target_path=target_path,
#         filtered_path=filtered_path,
#     )


# def _show_and_choose(
#     row,
#     config,
#     df,
#     index,
#     mode,
#     face_set,
#     df_full,
#     df_to_check_path,
#     target_path,
#     filtered_path,
# ):
#     """
#     Einheitliche Anzeige und Button‑Logik für beide Modi.
#     mode='relabel' oder 'normal'
#     """
#     imgfile = os.path.basename(row["file_name"])
#     imgpath = os.path.join(config.directories.images_target_dir, imgfile)
#     img = cv2.imread(imgpath)
#     if img is None:
#         st.error(f"Bild nicht gefunden: {imgpath}")
#         # Index inkrementieren
#         if mode == "relabel":
#             st.session_state.r_idx += 1
#         else:
#             st.session_state.current_index += 1
#         return

#     st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=imgfile)

#     if mode == "relabel":
#         total = len(st.session_state.to_relabel)
#         st.write(f"Nachlabeln {st.session_state.r_idx+1} / {total}")
#         a, p, s = st.columns(3)
#         if a.button("Animals"):
#             df.at[index, "relabel"] = "animals"
#         if p.button("Personen"):
#             df.at[index, "relabel"] = "personen"
#         if s.button("Skip"):
#             df.at[index, "relabel"] = "Skipped"
#         if df.at[index, "relabel"] is not None:
#             st.session_state.r_idx += 1
#         return

#     # normaler Label‑Modus
#     st.write(
#         f"Bild {st.session_state.current_index+1} / {len(st.session_state.labels)}"
#     )
#     labels = st.session_state.labels
#     if face_set:
#         st.markdown(f"**Accepted:** {sum(l=='Accepted' for l in labels)}")
#     else:
#         st.markdown(
#             f"**Animals:** {sum(l=='animals' for l in labels)}  |  "
#             f"**Personen:** {sum(l=='personen' for l in labels)}"
#         )

#     if face_set:
#         btn1, btn2 = st.columns(2)
#         if btn1.button("Accept"):
#             st.session_state.labels[index] = "Accepted"
#         if btn2.button("Skip"):
#             st.session_state.labels[index] = "Skipped"
#     else:
#         a, p, s = st.columns(3)
#         if a.button("Animals"):
#             st.session_state.labels[index] = "animals"
#         if p.button("Personen"):
#             st.session_state.labels[index] = "personen"
#         if s.button("Skip"):
#             st.session_state.labels[index] = "Skipped"

#     if st.session_state.labels[index] is not None:
#         # Sofort speichern im Hintergrund
#         df["label"] = st.session_state.labels
#         save_full_and_filtered(df_full, target_path, filtered_path)
#         # Index für nächstes Bild
#         st.session_state.current_index += 1


# if __name__ == "__main__":
#     st.title("Image Labeler")
#     config = load_config("config.yaml")
#     faces_set = False  # True = Face‑Modus, False = Non‑Face
#     subdir = "subj_01"  # fest auf subj_01 gesetzt

#     # Pfade prüfen
#     if faces_set:
#         df_to_check = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_face_nans_removed,
#         )
#         target = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_face_labelled,
#         )
#     else:
#         df_to_check = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_non_face_nans_removed,
#         )
#         target = os.path.join(
#             config.directories.excel_files_target_dir,
#             subdir,
#             config.dataset_creation.subset_animate_non_face_labelled,
#         )

#     label_images(config, df_to_check, target, faces_set)

# import streamlit as st
# import pandas as pd
# import cv2
# import os
# from utils.config import Configuration, load_config


# def save_full_and_filtered(df_full, target_full, filtered_full):
#     df_full.to_excel(target_full, index=False)
#     df_full[df_full["label"] == "animals"].to_excel(filtered_full, index=False)
#     st.success(f"Saved:\n • {target_full}\n • {filtered_full}")


# def label_images(
#     config: Configuration, df_to_check_path: str, target_path: str, face_set: bool
# ):
#     # 1) Data & paths
#     df_full = pd.read_excel(df_to_check_path)
#     base, ext = os.path.splitext(target_path)
#     filtered_path = f"{base}_filtered{ext}"

#     # 2) Session‑State initialisieren (nur einmal)
#     if "labels" not in st.session_state:
#         # Lade vorhandene Labels, behalte nur gültige Werte, sonst None
#         if os.path.exists(target_path):
#             prev = pd.read_excel(target_path)
#             raw_labels = prev["label"].tolist() if "label" in prev.columns else []
#         else:
#             raw_labels = []

#         valid = ("animals", "personen", "Skipped")
#         labels = [lab if lab in valid else None for lab in raw_labels]
#         if len(labels) != len(df_full):
#             labels = [None] * len(df_full)

#         # Finde erstes ungelabeltes Bild
#         start_idx = next(
#             (i for i, lab in enumerate(labels) if lab is None), len(labels)
#         )

#         st.session_state.labels = labels
#         st.session_state.current_index = start_idx

#     # 3) Abbruch, wenn alles gelabelt ist
#     if st.session_state.current_index >= len(df_full):
#         st.write("✅ Alle Bilder fertig gelabelt!")
#         df_full["label"] = st.session_state.labels
#         save_full_and_filtered(df_full, target_path, filtered_path)
#         return

#     # 4) Bild und Statistik anzeigen
#     idx = st.session_state.current_index
#     row = df_full.iloc[idx]
#     imgfile = os.path.basename(row["file_name"])
#     imgpath = os.path.join(config.directories.images_target_dir, imgfile)
#     img = cv2.imread(imgpath)
#     if img is None:
#         st.error(f"Bild nicht gefunden: {imgpath}")
#         st.session_state.current_index += 1
#         return  # nächster Durchlauf zeigt das nächste Bild

#     st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=imgfile)
#     st.write(f"Bild {idx+1} / {len(df_full)}")

#     # Live‑Stats
#     labels = st.session_state.labels
#     st.markdown(
#         f"**Animals:** {sum(l == 'animals' for l in labels)}  |  "
#         f"**Personen:** {sum(l == 'personen' for l in labels)}  |  "
#         f"**Skipped:** {sum(l == 'Skipped' for l in labels)}"
#     )

#     # Falls schon gelabelt, zeige das Label
#     current = st.session_state.labels[idx]
#     if current is not None:
#         st.markdown(f"**Aktuelles Label:** {current}")

#     # 5) Callback zum Labeln
#     def on_label(chosen: str):
#         st.session_state.labels[idx] = chosen
#         df_full["label"] = st.session_state.labels
#         save_full_and_filtered(df_full, target_path, filtered_path)
#         st.session_state.current_index += 1
#         # kein explizites rerun nötig, Button‑Click löst automatisch Neuladen aus

#     # 6) Buttons
#     a, p, s = st.columns(3)
#     a.button("Animals", on_click=on_label, args=("animals",))
#     p.button("Personen", on_click=on_label, args=("personen",))
#     s.button("Skip", on_click=on_label, args=("Skipped",))


# if __name__ == "__main__":
#     st.set_page_config(page_title="Image Labeler")
#     st.title("Image Labeler")
#     config = load_config("config.yaml")
#     faces_set = False
#     subdir = "subj_01"

#     df_to_check = os.path.join(
#         config.directories.excel_files_target_dir,
#         subdir,
#         config.dataset_creation.subset_animate_non_face_nans_removed,
#     )
#     target = os.path.join(
#         config.directories.excel_files_target_dir,
#         subdir,
#         config.dataset_creation.subset_animate_non_face_labelled,
#     )

#     label_images(config, df_to_check, target, faces_set)


import streamlit as st
import pandas as pd
import cv2
import os
import ast
from utils.config import Configuration, load_config


def save_full_and_filtered(df_full, target_full, filtered_full):
    df_full.to_excel(target_full, index=False)
    df_full[df_full["label"].isin(["animals", "personen"])].to_excel(
        filtered_full, index=False
    )
    st.success(f"Saved:\n • {target_full}\n • {filtered_full}")


def label_images(
    config: Configuration, df_to_check_path: str, target_path: str, face_set: bool
):
    # 1) Load dataframe & determine paths
    df_full = pd.read_excel(df_to_check_path)
    base, ext = os.path.splitext(target_path)
    filtered_path = f"{base}_filtered{ext}"

    # Predefined COCO animal classes
    animal_classes = {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "person",
    }

    # 2) Initialize session state (once)
    if "labels" not in st.session_state:
        # Load existing labels if available
        if os.path.exists(target_path):
            prev = pd.read_excel(target_path)
            raw_labels = prev["label"].tolist() if "label" in prev.columns else []
        else:
            raw_labels = []

        valid = ("animals", "personen", "Skipped")
        labels = [lab if lab in valid else None for lab in raw_labels]
        if len(labels) != len(df_full):
            labels = [None] * len(df_full)

        # Find first unlabelled image
        start_idx = next((i for i, lab in enumerate(labels) if lab is None), 0)
        print(f"{start_idx=}")

        st.session_state.labels = labels
        st.session_state.current_index = start_idx

    # # 3) Loop: skip non-fitting images when face_set=False
    # while True:
    #     idx = st.session_state.current_index
    #     if idx >= len(df_full):
    #         break
    #     labels_list = df_full.iloc[idx]["labels"]
    #     # If not labeling faces and no animal class present, skip

    #     intersection = list(set(labels_list) & animal_classes)
    #     print(f"{labels_list}\t{animal_classes}\t{intersection}\t{face_set}")
    #     if not face_set and len(intersection) == 0:

    #         st.session_state.current_index += 1
    #         continue
    #     # Otherwise, stop at a valid image
    #     break

    # 3) Determine next valid index (skip non-animal when face_set=False)
    idx = st.session_state.current_index
    n = len(df_full)
    if not face_set:
        found = None
        for i in range(idx, n):
            raw = df_full.iloc[i]["labels"]
            labels_list = raw if isinstance(raw, list) else ast.literal_eval(raw)
            if any(lbl in animal_classes for lbl in labels_list):
                found = i
                break
        if found is None:
            # no more valid images
            st.write("✅ All images have been labelled!")
            df_full["label"] = st.session_state.labels
            save_full_and_filtered(df_full, target_path, filtered_path)
            return
        st.session_state.current_index = found
        idx = found

    # 4) Finish if done
    if st.session_state.current_index >= len(df_full):
        st.write("✅ All images have been labelled!")
        df_full["label"] = st.session_state.labels
        save_full_and_filtered(df_full, target_path, filtered_path)
        return

    # 5) Display image and stats
    row = df_full.iloc[st.session_state.current_index]
    imgfile = os.path.basename(row["file_name"])
    imgpath = os.path.join(config.directories.images_target_dir, imgfile)
    img = cv2.imread(imgpath)
    if img is None:
        st.error(f"Image not found: {imgpath}")
        st.session_state.current_index += 1
        return

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=imgfile)
    st.write(f"Image {st.session_state.current_index+1} / {len(df_full)}")

    # Live stats
    stats = st.session_state.labels
    st.markdown(
        f"**Animals:** {sum(l == 'animals' for l in stats)}  |  "
        f"**Personen:** {sum(l == 'personen' for l in stats)}  |  "
        f"**Skipped:** {sum(l == 'Skipped' for l in stats)}"
    )

    # Show current label if exists
    current = st.session_state.labels[st.session_state.current_index]
    if current is not None:
        st.markdown(f"**Current Label:** {current}")

    # 6) Navigation buttons
    nav_col1, nav_col2 = st.columns(2)

    def go_previous():
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1

    def go_next():
        if st.session_state.current_index < len(df_full) - 1:
            st.session_state.current_index += 1

    nav_col1.button("Previous", on_click=go_previous)
    nav_col2.button("Next", on_click=go_next)

    # 7) Label callbacks
    def on_label(chosen: str):
        idx = st.session_state.current_index
        st.session_state.labels[idx] = chosen
        df_full["label"] = st.session_state.labels
        save_full_and_filtered(df_full, target_path, filtered_path)
        st.session_state.current_index += 1

    # 8) Label buttons
    a, p, s = st.columns(3)
    a.button("Animals", on_click=on_label, args=("animals",))
    p.button("Personen", on_click=on_label, args=("personen",))
    s.button("Skip", on_click=on_label, args=("Skipped",))


if __name__ == "__main__":
    st.set_page_config(page_title="Image Labeler")
    st.title("Image Labeler")
    config = load_config("config.yaml")
    faces_set = False
    subdir = "subj_01"

    df_to_check = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_non_face_nans_removed,
    )
    target = os.path.join(
        config.directories.excel_files_target_dir,
        subdir,
        config.dataset_creation.subset_animate_non_face_labelled,
    )

    label_images(config, df_to_check, target, faces_set)
