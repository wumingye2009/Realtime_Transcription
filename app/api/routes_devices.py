from fastapi import APIRouter, Depends

from app.models.audio import AvailableDevicesResponse
from app.services.audio.device_discovery import DeviceDiscoveryService


router = APIRouter(prefix="/api/devices", tags=["devices"])


def get_device_service() -> DeviceDiscoveryService:
    return DeviceDiscoveryService()


@router.get("", response_model=AvailableDevicesResponse)
async def list_devices(
    service: DeviceDiscoveryService = Depends(get_device_service),
) -> AvailableDevicesResponse:
    return service.list_devices()
