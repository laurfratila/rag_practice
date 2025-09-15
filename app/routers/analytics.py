from datetime import date
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.db.session import get_db

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/claims/monthly")
def claims_monthly(
    product_type: Optional[str] = Query(None, description="Filter by product_type (e.g., 'auto', 'homeowners')"),
    peril: Optional[str] = Query(None, description="Filter by peril (e.g., 'hail', 'flood')"),
    start: Optional[date] = Query(None, description="Include rows with loss_date >= start"),
    end: Optional[date] = Query(None, description="Include rows with loss_date < end (exclusive)"),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Returns monthly claim counts (chart-ready).
    """
    conditions = ["loss_date IS NOT NULL"]
    params = {}

    if product_type:
        conditions.append("product_type = :product_type")
        params["product_type"] = product_type
    if peril:
        conditions.append("peril = :peril")
        params["peril"] = peril
    if start:
        conditions.append("loss_date >= :start")
        params["start"] = start
    if end:
        conditions.append("loss_date < :end")
        params["end"] = end

    where = " AND ".join(conditions)
    sql = text(f"""
        SELECT
          date_trunc('month', loss_date)::date AS month,
          COUNT(*)::bigint AS claims
        FROM ins.claims
        WHERE {where}
        GROUP BY 1
        ORDER BY 1
    """)

    rows = db.execute(sql, params).mappings().all()
    # Return YYYY-MM-01 format for frontend convenience
    return [{"month": r["month"].isoformat(), "claims": int(r["claims"])} for r in rows]


@router.get("/claims/by-county")
def claims_by_county(
    month: date = Query(..., description="Any date within the month you want (e.g., 2025-06-01)"),
    product_type: Optional[str] = Query(None),
    peril: Optional[str] = Query(None),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """
    Returns counts of claims for the given calendar month grouped by county_code.
    """
    # Compute month boundaries [first_day, first_day_next_month)
    params = {"first": month.replace(day=1)}
    sql_bounds = text("SELECT (date_trunc('month', :first))::date AS start, (date_trunc('month', :first) + INTERVAL '1 month')::date AS finish")
    bounds = db.execute(sql_bounds, params).mappings().first()
    if not bounds:
        raise HTTPException(status_code=400, detail="Could not compute month bounds")

    conditions = ["loss_date >= :start", "loss_date < :finish"]
    params.update({"start": bounds["start"], "finish": bounds["finish"]})

    if product_type:
        conditions.append("product_type = :product_type")
        params["product_type"] = product_type
    if peril:
        conditions.append("peril = :peril")
        params["peril"] = peril

    where = " AND ".join(conditions)
    sql = text(f"""
        SELECT
          COALESCE(le.county_code, 'UNKNOWN') AS county_code,
          COUNT(*)::bigint AS claims
        FROM ins.claims c
        JOIN ins.loss_events le ON le.claim_id = c.claim_id
        WHERE {where}
        GROUP BY COALESCE(le.county_code, 'UNKNOWN')
        ORDER BY claims DESC NULLS LAST, county_code
        """)


    rows = db.execute(sql, params).mappings().all()
    return [{"county_code": r["county_code"], "claims": int(r["claims"])} for r in rows]
