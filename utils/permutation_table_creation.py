import pandas as pd
import numpy as np
from IPython.display import display
import dataframe_image as dfi

# Import matplotlib components for custom coloring
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import to_hex, to_rgba # Import to_rgba to get RGBA values

# Define a custom function for coloring p-values
def color_p_values_log10_custom(s, vmin_log10=0, vmax_log10=5, cmap_name='YlGnBu', epsilon=1e-300):
    """
    Applies background gradient based on -log10 transformation of p-values.
    Also sets font color to white or black for readability.
    's' is a Series (single column passed by apply) or DataFrame slice (if axis=None used by apply).
    Returns a Series/DataFrame of CSS style strings.
    """
    numeric_s = s.replace([np.inf, -np.inf], np.nan)

    log_p_values = -np.log10(numeric_s.clip(lower=epsilon))

    cmap = plt.colormaps.get_cmap(cmap_name) # Modern way to get colormap
    norm = plt.Normalize(vmin=vmin_log10, vmax=vmax_log10)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Helper function to determine text color based on background RGBA
    def get_text_color(rgba_color):
        # Calculate perceived luminance (using ITU-R BT.709 coefficients for sRGB)
        # Luminance is a measure of perceived brightness.
        # Values are typically 0-1. A common threshold for light/dark is 0.5.
        r, g, b, _ = rgba_color # ignore alpha channel for text color decision
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return 'white' if luminance < 0.5 else 'black' # Use white text on dark backgrounds, black on light

    # Function to apply color and text style for a single cell value
    def apply_single_cell_style(x):
        if pd.notna(x):
            rgba_color = mapper.to_rgba(x)
            hex_color = to_hex(rgba_color)
            text_color = get_text_color(rgba_color)
            return f'background-color: {hex_color}; color: {text_color};'
        return '' # Return empty string for NaN values (no styling)

    if isinstance(s, pd.Series):
        return log_p_values.apply(apply_single_cell_style)
    elif isinstance(s, pd.DataFrame):
        styled_df = pd.DataFrame('', index=s.index, columns=s.columns)
        for col in log_p_values.columns:
            styled_df[col] = log_p_values[col].apply(apply_single_cell_style)
        return styled_df
    else:
        return s.apply(lambda x: '') # Fallback for unexpected input type


df = pd.read_excel("../data/neural_net_results/subj_01/results.xlsx")

for module in ["detection" , "genderage"]:
    df_rdm = df[df['distance_space']=='RDM']
    df_rdm = df_rdm[df_rdm['module']==module]

    layers = df_rdm["layer"].to_list()
    seen = set()
    seen_add = seen.add
    layer_order = [x for x in layers if not (x in seen or seen_add(x))]

    pivot = df_rdm.pivot_table(
        index='layer',
        columns='feature_selection',
        values=['r_obs','p_value']
    )

    # Correct way to rename the top-level column names without reordering levels
    new_level0_names = []
    for col_name in pivot.columns.levels[0]:
        if col_name == 'r_obs':
            new_level0_names.append('Effect size')
        elif col_name == 'p_value':
            new_level0_names.append('p-value')
        else:
            new_level0_names.append(col_name)

    pivot.columns = pivot.columns.set_levels(new_level0_names, level=0)


    # name the column levels so it’s clearer in the HTML/LaTeX:
    pivot.columns.set_names(['Metric','Feature'], inplace=True)

    # re‐order your layers:
    pivot = pivot.reindex(layer_order)

    # 2) Style with grouped headers and correct subset slicing
    idx = pd.IndexSlice
    styled = (
        pivot.style
            .format(precision=3)
            # effect‐size heatmap on Effect size columns
            # background_gradient usually handles text color automatically for this one
            .background_gradient(
                cmap='coolwarm',
                subset=(idx[:], idx['Effect size', :]),
                vmin=-pivot['Effect size'].abs().max().max(),
                vmax= pivot['Effect size'].abs().max().max(),
            )
            # Apply custom p-value coloring function with built-in text color logic
            .apply(
                color_p_values_log10_custom,
                subset=(idx[:], idx['p-value', :]),
                axis=0, # Apply column by column
                vmin_log10=0,
                vmax_log10=10,
                cmap_name='YlGnBu'
            )
            .set_table_styles([
                {'selector':'th, td',
                'props':[
                    ('padding','2px 4px'),
                    ('font-size','8pt')
                ]}
            ])
    )


    # 1) Add explicit table‐and‐cell borders (and collapse them)
    styled_borders = (
        styled
        .set_table_styles([
            # collapse borders and make table full width
            {'selector': 'table',
            'props': [
                ('border-collapse', 'collapse'),
                ('width', '100%'),
            ]},
            # border on all th and td
            {'selector': 'th, td',
            'props': [
                ('border', '1px solid #888'),
                ('padding', '4px'),
            ]},
        ])
    )


    # now this won’t crash
    dfi.export(
        styled_borders,
        f"../data/plots/thesis/nn_feature_encoding_{module}.png",
        table_conversion='playwright',
        dpi=800
    )


# -----------------------------------------
# import pandas as pd
# from IPython.display import display
# import dataframe_image as dfi

# df = pd.read_excel("data/neural_net_results/subj_01/results.xlsx")

# for module in ["detection" , "genderage"]:
#     df_rdm = df[df['distance_space']=='RDM']
#     df_rdm = df_rdm[df_rdm['module']==module]

#     layers = df_rdm["layer"].to_list()
#     seen = set()
#     seen_add = seen.add
#     layer_order = [x for x in layers if not (x in seen or seen_add(x))]

#     pivot = df_rdm.pivot_table(
#         index='layer',
#         columns='feature_selection',
#         values=['r_obs','p_value']
#     )

#     # Correct way to rename the top-level column names without reordering levels
#     # Create a new list of level 0 (Metric) names based on the existing ones
#     new_level0_names = []
#     for col_name in pivot.columns.levels[0]:
#         if col_name == 'r_obs':
#             new_level0_names.append('Effect size')
#         elif col_name == 'p_value':
#             new_level0_names.append('p-value')
#         else:
#             new_level0_names.append(col_name) # Keep other names if any

#     # Assign the new levels directly
#     pivot.columns = pivot.columns.set_levels(new_level0_names, level=0)


#     # name the column levels so it’s clearer in the HTML/LaTeX:
#     pivot.columns.set_names(['Metric','Feature'], inplace=True)

#     # re‐order your layers:
#     pivot = pivot.reindex(layer_order)

#     # 2) Style with grouped headers and correct subset slicing
#     idx = pd.IndexSlice
#     styled = (
#         pivot.style
#             .format(precision=3)
#             # effect‐size heatmap on Effect size columns
#             .background_gradient(
#                 cmap='coolwarm',
#                 subset=(idx[:], idx['Effect size', :]),
#                 vmin=-pivot['Effect size'].abs().max().max(),
#                 vmax= pivot['Effect size'].abs().max().max(),
#             )
#             # p‐value heatmap on p-value columns
#             .background_gradient(
#                 cmap='YlGnBu_r',
#                 subset=(idx[:], idx['p-value', :]),
#                 vmin=0, vmax=1
#             )
#             .set_table_styles([
#                 {'selector':'th, td',
#                 'props':[
#                     ('padding','2px 4px'),
#                     ('font-size','8pt')
#                 ]}
#             ])
#     )


#     # 1) Add explicit table‐and‐cell borders (and collapse them)
#     styled_borders = (
#         styled
#         .set_table_styles([
#             # collapse borders and make table full width
#             {'selector': 'table',
#             'props': [
#                 ('border-collapse', 'collapse'),
#                 ('width', '100%'),
#             ]},
#             # border on all th and td
#             {'selector': 'th, td',
#             'props': [
#                 ('border', '1px solid #888'),
#                 ('padding', '4px'),
#             ]},
#         ])
#     )


#     # now this won’t crash
#     dfi.export(
#         styled_borders,
#         f"data/plots/thesis/nn_feature_encoding_{module}.png",
#         table_conversion='playwright',
#         dpi=800
#     )


# import pandas as pd
# from IPython.display import display
# import dataframe_image as dfi

# df = pd.read_excel("data/neural_net_results/subj_01/results.xlsx")

# for module in ["detection" , "genderage"]:
#     df_rdm = df[df['distance_space']=='RDM']
#     df_rdm = df_rdm[df_rdm['module']==module]

#     layers = df_rdm["layer"].to_list()
#     seen = set()
#     seen_add = seen.add
#     layer_order = [x for x in layers if not (x in seen or seen_add(x))]

#     pivot = df_rdm.pivot_table(
#         index='layer',
#         columns='feature_selection',
#         values=['r_obs','p_value']
#     )

#     # 1) Name the column levels first
#     pivot.columns.set_names(['Metric','Feature'], inplace=True)

#     # 2) Now, set the levels using the named 'Metric' level
#     pivot.columns = pivot.columns.set_levels(['Effect size', 'p-value'], level='Metric')

#     # re‐order your layers:
#     pivot = pivot.reindex(layer_order)

#     # 2) Style with grouped headers and correct subset slicing
#     idx = pd.IndexSlice
#     styled = (
#         pivot.style
#             .format(precision=3)
#             # effect‐size heatmap on Effect size columns
#             .background_gradient(
#                 cmap='coolwarm',
#                 subset=(idx[:], idx['Effect size', :]),
#                 vmin=-pivot['Effect size'].abs().max().max(),
#                 vmax= pivot['Effect size'].abs().max().max(),
#             )
#             # p‐value heatmap on p-value columns
#             .background_gradient(
#                 cmap='YlGnBu_r',
#                 subset=(idx[:], idx['p-value', :]),
#                 vmin=0, vmax=1
#             )
#             .set_table_styles([
#                 {'selector':'th, td',
#                 'props':[
#                     ('padding','2px 4px'),
#                     ('font-size','8pt')
#                 ]}
#             ])
#     )

#     # 1) Add explicit table‐and‐cell borders (and collapse them)
#     styled_borders = (
#         styled
#         .set_table_styles([
#             # collapse borders and make table full width
#             {'selector': 'table',
#             'props': [
#                 ('border-collapse', 'collapse'),
#                 ('width', '100%'),
#             ]},
#             # border on all th and td
#             {'selector': 'th, td',
#             'props': [
#                 ('border', '1px solid #888'),
#                 ('padding', '4px'),
#             ]},
#         ])
#     )

#     # now this won’t crash
#     dfi.export(
#         styled_borders,
#         f"data/plots/thesis/nn_feature_encoding_{module}.png",
#         table_conversion='playwright',
#         dpi=800
#     )
