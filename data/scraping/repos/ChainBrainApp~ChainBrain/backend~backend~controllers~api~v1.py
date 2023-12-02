import openai


from backend.services.openAI.service import OpenAIService
from backend.services.graph.service import GraphService
from backend.services.dashboard.service import DashboardService
from backend.models.dashboard import DashboardQueryResult
from backend.services.ceramic.service import CeramicService


def QUERY_API_RESPONSE_FORMATTER(id, chatgpt_gql, output):
    return {"id": id, "chatgpt_gql": chatgpt_gql, "output": output}


class APIV1Controller:
    def handle_query_for_dashboard(self, input_sentence, subgraph, wallet_address):
        """Get data for query

        Parameters:
        ----------
        input_sentence : _type_
        """
        ai_service = OpenAIService()
        gql = ai_service.request_gql_for_graph_llama(input_sentence, subgraph)
        graph_service = GraphService(protocol=subgraph)
        try:
            result = graph_service.query_thegraph(gql)
        except ValueError:
            # import pdb;pdb.set_trace()
            return QUERY_API_RESPONSE_FORMATTER("-1", gql, [])

        dashboard_query_result = DashboardQueryResult(
            user_input=input_sentence,
            subgraph=subgraph,
            chatgpt_gql=str(gql),
            output=result,
            gql_valid=-1,
            user_id=wallet_address,
        )

        DashboardService().save_dashboard_query_result(dashboard_query_result)
        return QUERY_API_RESPONSE_FORMATTER(dashboard_query_result.id, gql, result)

    def get_dashboard(self, dashboard_id):
        return DashboardQueryResult.query.get(dashboard_id).to_dict()

    def get_dashboards(self, wallet_address):
        return [
            board.to_dict()
            for board in DashboardQueryResult.query.filter_by(
                user_id=wallet_address
            ).all()
        ]

    def save_dashboard_to_user(self, dashboard_id, wallet_address):
        dashboard = DashboardQueryResult.query.get(dashboard_id)
        dashboard.user_id = wallet_address
        DashboardService().save_dashboard_query_result(dashboard)
        CeramicService().save_feedback_with_query(dashboard)

    def save_dashboard_feedback(self, dashboard_id, feedback):
        dashboard =  DashboardQueryResult.query.get(dashboard_id)
        dashboard.gql_valid = feedback
        DashboardService().save_dashboard_query_result(dashboard)
