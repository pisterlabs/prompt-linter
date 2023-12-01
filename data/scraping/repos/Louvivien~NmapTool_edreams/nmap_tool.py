from superagi.tools.base_tool import BaseTool 
from pydantic import BaseModel, Field
from typing import Type  

import nmap
from superagi.llms.openai import OpenAi
from superagi.config.config import get_config  # Importación de get_config

class NmapInput(BaseModel):
    target: str = Field(..., description="Target IP or range")

class NmapTool(BaseTool):

    #name = "Nmap Scanner"
    #args_schema = NmapInput
    #description = "Nmap network scanner with automated analysis"
    name: str = "Nmap Scan Tool"
    args_schema: Type[BaseModel] = NmapInput
    description: str = "Performs a basic Nmap scan on a IP"
    
    def _execute(self, target: str) -> str:
        scan_result = self.scan(target)
        findings = self.process_findings(scan_result, target) 
        summary = self.generate_summary(findings, target)
        #return summary + "\n\n" + scan_result
        return f"{summary}\n\n{str(scan_result)}"  # Convertir scan_result en una cadena antes de concatenar

    def scan(self, target):
        # Ejecutar Nmap 
        nm = nmap.PortScanner()
        scan_args = '-sV '
        result = nm.scan(target, arguments=scan_args)

        # Devolver raw result
        return result 

    def process_findings(self, scan_result, target):
        try:
            # Verificar si los resultados están disponibles para el objetivo
            if target not in scan_result['scan']:
                return None  # Devolver None si no hay resultados para el objetivo

            # Extraer puertos abiertos
            open_ports = scan_result['scan'][target]['tcp'].keys()

            # Extraer sistema operativo (si está disponible)
            os = "Unknown OS"  # Valor predeterminado en caso de que no se encuentre el sistema operativo
            if 'osmatch' in scan_result['scan'][target]:
                os_match = scan_result['scan'][target]['osmatch'][0]
                os = os_match['name']

            # Extraer versiones de servicios (si están disponibles)
            services = {}
            if 'tcp' in scan_result['scan'][target]:
                for port in open_ports:
                    service = scan_result['scan'][target]['tcp'][port]
                    services[port] = service['name'] + " " + service['version']

            # Devolver diccionario de hallazgos
            findings = {
                "open_ports": open_ports,
                "os": os,
                "services": services
            }

            return findings

        except KeyError:
            return None  # Devolver None si hay una KeyError (campo faltante) en los resultados

    def generate_summary(self, findings, target):
        # Construir prompt con plantilla
        prompt = f"Here are the key findings from scanning the IP address {target}:\n\nOpen Ports: {findings['open_ports']}\n\nOperating System: {findings['os']}\n\nServices:\n{findings['services']}\n\nPlease generate a detailed security analysis summary in markdown format for the CISO covering:\n- Overview of findings\n- Potential security implications\n- Next steps and recommendations"
        
        # Crear instancia de la clase OpenAi con la clave de API adecuada
        api_key = get_config('OPENAI_API_KEY')  # Obtener clave de API
        openai_instance = OpenAi(api_key=api_key)
        
        # Llamar a chat_completion de OpenAi
        response = openai_instance.chat_completion(messages=[{"role": "system", "content": "You are a helpful assistant that provides security analysis."}, {"role": "user", "content": prompt}])
        
        # Extraer el contenido del resultado
        summary = response['content']

        return summary
