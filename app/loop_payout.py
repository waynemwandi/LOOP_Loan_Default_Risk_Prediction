import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import requests

from app.loop_auth import LoopAuth

logger = logging.getLogger("loop-risk.payout")


def _now_tick() -> str:
    # 20250918102110112 style: YYYYMMDDhhmmssSSS
    # return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    return datetime.now().strftime("%Y%m%d%H%M%S")


class LoopPayout:
    def __init__(self, auth: LoopAuth):
        self.auth = auth
        self.base = (os.getenv("LOOP_PESALINK_BASE") or "").rstrip("/")
        self.verify_tls = os.getenv("LOOP_VERIFY_TLS", "true").lower() == "true"
        # ONE char numeric flag expected by this endpoint: '1' (yes) / '0' (no)
        self.is_send_advice = (
            "1"
            if str(os.getenv("LOOP_ADVICE", "Y")).strip().lower()
            in {"y", "yes", "true", "1"}
            else "0"
        )

        if not self.base:
            raise ValueError("LOOP_PESALINK_BASE is not set")

        # sender / merchant
        self.partner_id = os.getenv("LOOP_PARTNER_ID", "LOOP")
        self.product_set = os.getenv("LOOP_PRODUCT_SET_CODE", "")
        self.sender_usr_no = os.getenv("LOOP_SENDER_USR_NO", "")
        self.sender_usr_type = os.getenv("LOOP_SENDER_USRTYPE", "8")  # 8=merchant
        self.sender_instrument = os.getenv("LOOP_SENDER_INSTRUMENT", "0012")  # bank

        # transfer defaults
        self.transfer_channel = os.getenv("LOOP_TRANSFER_CHANNEL", "65")
        self.transfer_type = os.getenv("LOOP_TRANSFER_TYPE", "20")
        self.ccy = os.getenv("LOOP_CURRENCY", "KES")
        self.default_bank_code = os.getenv("LOOP_DEFAULT_BANK_CODE", "")
        self.purpose = os.getenv("LOOP_PURPOSE", "TRF")
        self.fee_amount = os.getenv("LOOP_PAYOUT_FEE", "10.0")
        self.tax_amount = os.getenv("LOOP_PAYOUT_TAX", "1.5")

        self.dry_run = os.getenv("LOOP_PAYOUT_DRY_RUN", "true").lower() == "true"

    def _post(self, url: str, payload: dict) -> dict:
        headers = {
            "Authorization": self.auth.get_token(),
            "Content-Type": "application/json",
        }
        full_url = (
            url
            if url.startswith("http")
            else f"{self.base.rstrip('/')}/{url.lstrip('/')}"
        )
        try:
            r = requests.post(
                full_url,
                json=payload,
                headers=headers,
                timeout=20,
                verify=self.verify_tls,
            )
        except requests.RequestException:
            logger.exception("POST %s network error", full_url)
            raise

        body_preview = (r.text or "")[:1000]
        logger.info(
            "LOOP %s -> HTTP %s body=%s",
            full_url.split("/")[-1],
            r.status_code,
            body_preview,
        )

        try:
            r.raise_for_status()
        except requests.HTTPError:
            raise RuntimeError(f"LOOP {full_url} HTTP {r.status_code}: {body_preview}")

        try:
            return r.json()
        except ValueError:
            return {"_raw": body_preview}

    def _build_funds_payload(
        self,
        amount: float,
        recipient_mobile: str,
        recipient_account_type: Optional[str],
        recipient_bank_code: Optional[str],
        recipient_name: Optional[str],
        narration: str,
    ) -> dict:
        tick = _now_tick()
        return {
            "partnerId": self.partner_id,
            "productSetCode": self.product_set or "LOOP",
            "transferRefNo": tick,
            "transferType": self.transfer_type,
            "transactionDateTime": tick,
            "currency": self.ccy,
            "paymentAmount": f"{amount:.2f}",
            "feeAmount": f"{float(self.fee_amount):.2f}",
            "taxAmount": f"{float(self.tax_amount):.2f}",
            "sender": {
                "senderInstrument": self.sender_instrument,
                "usrType": self.sender_usr_type,
                "usrNo": self.sender_usr_no,
            },
            "recipient": {
                "recipientAccountType": recipient_account_type or "MOMO",
                "mobileNo": recipient_mobile,
                "bankCode": (recipient_bank_code or self.default_bank_code or "18"),
                "accountName": recipient_name or "",
                "recipientAddress": {"country": "Kenya"},
            },
            "transferChannel": self.transfer_channel,
            "isSendAdvice": self.is_send_advice,  # 'Y' or 'N'
            "purposeOfPayment": self.purpose,
            "remark": narration,
            "transactionReferenceNo": tick,
            "requestTime": tick,
            "requestId": tick,
            "service": "fundsTransfer",
        }

    # === Trinity-API-Onboarding ===
    def get_user_detail(self, msisdn: str) -> dict:
        payload = {"mobileNo": msisdn}
        return self._post(f"{self.base}/openapi/getUserDetail.djson", payload)

    # === Trinity-API-Send Money ===
    def payment_instruments_enquiry(
        self,
        pay_amount: Optional[str] = None,
        pay_way_type: Optional[str] = None,
        usr_no: Optional[str] = None,
        usr_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        # POST /openapi/paymentInstrumentsEnquiry.djson
        if self.dry_run:
            logger.info("[DRY RUN] payment_instruments_enquiry")
            return {
                "ok": True,
                "dry_run": True,
                "feeAmount": self.fee_amount,
                "taxAmount": self.tax_amount,
            }
        payload = {
            "partnerId": self.partner_id,
            "productSetCode": self.product_set,
            "usrNo": usr_no or self.sender_usr_no,
            "usrType": usr_type or self.sender_usr_type,
            "payAmount": pay_amount or "",
            "useOverdraft": "N",
            "payWayType": pay_way_type or "",
        }
        return self._post("/openapi/paymentInstrumentsEnquiry.djson", payload)

    def funds_transfer(
        self,
        *,
        amount: float,
        recipient_mobile: str,
        recipient_account_type: str,
        recipient_bank_code: str | None,
        recipient_name: str | None,
        narration: str,
    ) -> dict:
        # Respect DRY RUN from .env
        payload = self._build_funds_payload(
            amount,
            recipient_mobile,
            recipient_account_type,
            recipient_bank_code,
            recipient_name,
            narration,
        )
        if self.dry_run:
            logger.info(
                "[DRY RUN] funds_transfer payload=%s",
                {
                    **payload,
                    "paymentAmount": "***",
                    "feeAmount": "***",
                    "taxAmount": "***",
                },
            )
            return {
                "ok": True,
                "status": "DRY_RUN_OK",
                "amount": amount,
                "transferRefNo": payload["transferRefNo"],
                "transactionReferenceNo": payload["transactionReferenceNo"],
                "requestId": payload["requestId"],
                "transferType": payload["transferType"],
                "transferChannel": payload["transferChannel"],
                "raw": {"_dry_run": True},
            }
        logger.info("funds_transfer isSendAdvice=%s", payload.get("isSendAdvice"))
        data = self._post(f"{self.base}/openapi/fundsTransfer.djson", payload)
        data = self._post(f"{self.base}/openapi/fundsTransfer.djson", payload)

        # Normalize summary for the UI/template
        def _ok(d: dict) -> bool:
            # Accept common success codes across LOOP variants
            code = (
                (d.get("rspCode") or d.get("responseCode") or d.get("code") or "")
                .strip()
                .upper()
            )
            status = (d.get("status") or "").strip().upper()
            return code in {"0000", "00", "SUCCESS"} or status in {"SUCCESS", "OK"}

        return {
            "ok": _ok(data),
            "status": data.get("rspMessage")
            or data.get("responseDescription")
            or data.get("message")
            or data.get("status"),
            "amount": amount,
            "transferRefNo": data.get("transferRefNo") or payload["transferRefNo"],
            "transactionReferenceNo": data.get("transactionReferenceNo")
            or payload["transactionReferenceNo"],
            "requestId": data.get("requestId") or payload["requestId"],
            "transferType": data.get("transferType") or payload["transferType"],
            "transferChannel": data.get("transferChannel")
            or payload["transferChannel"],
            "raw": data,
        }
