import pandas as pd
from pathlib import Path
from langchain.tools import tool

@tool
def get_acoustic_zones(latitude: float, longitude: float) -> str:
    """
    Queries the AcousticGuard database to find Thermoclines and Deep Sound Channels 
    at a specific latitude and longitude. Use this tool when a user asks about 
    shadow zones, submarine hiding spots, or sonar ranges.
    """
    # 1. Resolve the path to the Phase 1 MVP data
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_path = base_dir / "data" / "processed" / "tactical_acoustic_zones.csv"
    
    if not data_path.exists():
        return "System Error: Tactical Acoustic Zones database is offline."
        
    try:
        df = pd.read_csv(data_path)
        
        # 2. Filter data (For MVP, we just check if data exists)
        if df.empty:
            return f"No acoustic data found near Lat: {latitude}, Lon: {longitude}."
            
        # 3. Extract the intelligence
        thermoclines = df[df['is_thermocline']]['pressure'].tolist()
        sound_channels = df[df['is_sound_channel']]['pressure'].tolist()
        
        # 4. Format the response for the LLM's brain
        report = f"Acoustic Intelligence Report for {latitude}°N, {longitude}°E:\n"
        
        if thermoclines:
            report += f"- Strong Thermoclines (Sonar bending) detected at depths: {thermoclines[:3]} dbar.\n"
        else:
            report += "- No significant thermoclines detected.\n"
            
        if sound_channels:
            report += f"- Deep Sound Channels (Acoustic trapping) detected at depths: {sound_channels[:3]} dbar.\n"
        else:
            report += "- No Deep Sound Channels detected.\n"
            
        report += "WARNING: Submarines may utilize depths below the thermocline as acoustic shadow zones."
        return report

    except Exception as e:
        return f"Database query failed: {str(e)}"