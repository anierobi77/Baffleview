import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import requests

class WitsmlLiveStreamClient:
    def __init__(self, server_url=None, token=None):
        self.url = server_url or "https://exxonmobil.com"
        self.headers = {"Authorization": f"Bearer {token or 'MOCK_RIG_TOKEN_772'}"}
        print(f"?? Secure WITSML Client pointing to: {self.url}")

    def fetch_live_lwd_packet(self):
        """
        Simulates fetching an XML data packet from a live rig WITSML server.
        In production, this runs a requests.get() or WebSockets listener loop.
        """
        # Mocking raw WITSML XML payload response from an active drilling bit
        mock_witsml_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <trajectorys xmlns="http://witsml.org">
            <trajectory>
                <idWell>East_Ford_3H</idWell>
                <idWellbore>Borehole_01</idWellbore>
                <trajectoryStation>
                    <md uom="ft">2742.5</md>
                    <tvd uom="ft">2734.1</tvd>
                    <incl uom="rad">1.553</incl>
                    <gammaRay uom="gAPI">128.4</gammaRay>
                    <resDeep uom="ohm.m">5.2</resDeep>
                    <porosity uom="%">6.2</porosity>
                </trajectoryStation>
            </trajectory>
        </trajectorys>
        """
        return mock_witsml_xml

    def parse_witsml_to_dataframe(self, xml_string):
        """Parses the raw WITSML structure directly into a clean Python dictionary."""
        root = ET.fromstring(xml_string)
        
        # Locate the station metrics within the standard schema layers
        namespaces = {'w': 'http://witsml.org'}
        station = root.find('.//w:trajectoryStation', namespaces)
        
        if station is not None:
            data_packet = {
                'Depth': float(station.find('w:md', namespaces).text),
                'TVD': float(station.find('w:tvd', namespaces).text),
                'GR': float(station.find('w:gammaRay', namespaces).text),
                'Resistivity': float(station.find('w:resDeep', namespaces).text),
                'Porosity': float(station.find('w:porosity', namespaces).text)
            }
            return data_packet
        return None
