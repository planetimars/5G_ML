"""PDF export helpers for experiment tables and figures."""

from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _table_to_reportlab(table_df, max_rows=20, max_cols=8):
    clipped = table_df.copy().head(max_rows)
    if clipped.shape[1] > max_cols:
        clipped = clipped.iloc[:, :max_cols]

    data = [list(clipped.columns)] + clipped.fillna("").astype(str).values.tolist()
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dcecff")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12385f")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#9fb6d1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return tbl


def _figure_to_image(fig):
    png_bytes = fig.to_image(format="png", width=1200, height=650, scale=2)
    img_buffer = BytesIO(png_bytes)
    image = Image(img_buffer)
    image._restrictSize(7.2 * inch, 4.6 * inch)
    return image


def build_pdf_report(title, tables=None, figures=None):
    """Build a PDF report from dictionaries of DataFrames and Plotly figures."""
    tables = tables or {}
    figures = figures or {}

    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20,
    )

    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]

    if tables:
        story.append(Paragraph("Tables", styles["Heading2"]))
        story.append(Spacer(1, 6))
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            story.append(Paragraph(str(name), styles["Heading3"]))
            story.append(_table_to_reportlab(df))
            story.append(Spacer(1, 10))

    if figures:
        story.append(PageBreak())
        story.append(Paragraph("Figures", styles["Heading2"]))
        story.append(Spacer(1, 6))
        for name, fig in figures.items():
            story.append(Paragraph(str(name), styles["Heading3"]))
            try:
                story.append(_figure_to_image(fig))
            except Exception as exc:
                story.append(Paragraph(f"Could not render figure: {exc}", styles["Normal"]))
            story.append(Spacer(1, 10))

    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
