from datetime import date
from io import BytesIO

from flask import Blueprint, Response, render_template, request, send_file

from ..security import login_required
from ..services.report_service import attendance_dataframe, dataframe_to_excel_bytes, summary_from_dataframe


bp = Blueprint("reports", __name__, url_prefix="/reports")


@bp.get("/")
@login_required
def index():
    period = _report_period()
    anchor_date = _report_date()
    dataframe, date_range = attendance_dataframe(period, anchor_date)
    summary = summary_from_dataframe(dataframe)
    records = dataframe.to_dict(orient="records") if not dataframe.empty else []
    return render_template(
        "reports/index.html",
        period=period,
        anchor_date=anchor_date,
        date_range=date_range,
        summary=summary,
        records=records,
    )


@bp.get("/export/csv")
@login_required
def export_csv():
    period = _report_period()
    anchor_date = _report_date()
    dataframe, _date_range = attendance_dataframe(period, anchor_date)
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    filename = f"attendance_{period}_{anchor_date.isoformat()}.csv"
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@bp.get("/export/excel")
@login_required
def export_excel():
    period = _report_period()
    anchor_date = _report_date()
    dataframe, _date_range = attendance_dataframe(period, anchor_date)
    excel_bytes = dataframe_to_excel_bytes(dataframe)
    filename = f"attendance_{period}_{anchor_date.isoformat()}.xlsx"
    return send_file(
        BytesIO(excel_bytes),
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _report_period() -> str:
    period = request.args.get("period", "daily").lower()
    return period if period in {"daily", "weekly", "monthly"} else "daily"


def _report_date() -> date:
    try:
        return date.fromisoformat(request.args.get("date", date.today().isoformat()))
    except ValueError:
        return date.today()
