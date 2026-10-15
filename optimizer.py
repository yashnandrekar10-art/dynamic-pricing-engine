import joblib
import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from typing import Dict , Any
import os

class DynamicPricingEngine:
    def __init__(self,model_name: str="demand_model.joblib"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                "Ensure you ran the export cell in eda.ipynb and the file is in this folder."
            )
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.feature_cols = artifact['feature_cols']

    def predict_demand(self, candidate_price: float, context: Dict[str, Any]) -> float:
        row = context.copy()
        row['unit_price'] = candidate_price
        row['landed_price'] = candidate_price + row['freight_price']
        row['price_diff_comp_1'] = candidate_price - row['comp_1']
        row['price_ratio_comp_1'] = candidate_price / max(row['comp_1'], 0.01)
        row['price_diff_comp_2'] = candidate_price - row['comp_2']
        row['price_ratio_comp_2'] = candidate_price / max(row['comp_2'], 0.01)

        X_input = pd.DataFrame([row])[self.feature_cols]
        pred_log = self.model.predict(X_input)[0]
        return float(np.exp(pred_log))

    def optimize_price(
        self,
        context: Dict[str, Any],
        unit_cost: float,
        stock_available: int,
        min_margin_pct: float = 0.05,
        max_price_multiplier: float = 1.30
    ) -> Dict[str, Any]:
        current_price = context['unit_price']
        comp_price = context['comp_1']

        # Baseline performance at current price tag
        baseline_qty = self.predict_demand(current_price, context)
        baseline_profit = (current_price - unit_cost) * baseline_qty

        # Objective Function: Minimize negative profit
        def objective(price_arr):
            p = price_arr[0]
            q = self.predict_demand(p, context)
            return -((p - unit_cost) * q)

        # 1. Minimum price: Never sell below cost + minimal margin
        p_min = max(unit_cost * (1 + min_margin_pct), comp_price * 0.70)
        # 2. Maximum price: Cap at 30% above current or competitor price
        p_max = max(current_price * max_price_multiplier, comp_price * max_price_multiplier)
        bounds = [(p_min, p_max)]

        stock_constraint = {
            'type': 'ineq',
            'fun': lambda p: stock_available - self.predict_demand(p[0], context)
        }

        # Run SLSQP Solver
        res = minimize(
            fun=objective,
            x0=[current_price],
            method='SLSQP',
            bounds=bounds,
            constraints=[stock_constraint]
        )

        optimal_price = round(float(res.x[0]), 2)
        optimal_qty = round(self.predict_demand(optimal_price, context), 1)
        optimal_profit = round(-float(res.fun), 2)
        profit_lift_pct = round(
            ((optimal_profit - baseline_profit) / baseline_profit) * 100, 2
        ) if baseline_profit > 0 else 0.0

        # Diagnosis logic (+/- 2% threshold)
        if optimal_price > current_price * 1.02:
            status = "UNDERPRICED"
            action = f"Raise price by ₹{optimal_price - current_price:.2f}"
        elif optimal_price < current_price * 0.98:
            status = "OVERPRICED"
            action = f"Cut price by ₹{current_price - optimal_price:.2f}"
        else:
            status = "OPTIMAL"
            action = "Maintain current price"

        return {
            "current_price": current_price,
            "recommended_price": optimal_price,
            "status": status,
            "action": action,
            "baseline_demand": round(baseline_qty, 1),
            "projected_demand": optimal_qty,
            "baseline_profit": round(baseline_profit, 2),
            "projected_profit": optimal_profit,
            "profit_lift_pct": profit_lift_pct
        }