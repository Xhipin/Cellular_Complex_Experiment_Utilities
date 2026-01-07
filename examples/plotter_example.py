from cellexp_util.plotter import Plotter
from cellexp_util.plotter import render_plot_job, PlotRenderConfig

plotter = Plotter("ablation.yaml")
for job_data in plotter.iter_loaded():
    render_plot_job(
        job_data,
        config=PlotRenderConfig(output_dir="experiments/plots", formats=("pdf", "png")),
    )


from cellexp_util.plotter import TableMaker
from cellexp_util.plotter import render_table_job, TableRenderConfig

tables = TableMaker("comparison.yaml")
for job_data in tables.iter_loaded():
    render_table_job(
        job_data,
        config=TableRenderConfig(output_dir="experiments/plots", formats=("md", "tex")),
    )
