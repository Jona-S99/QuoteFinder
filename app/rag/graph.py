# ---------------------------------------------------------------------------
# Grafo de Langgraph
# - Para construir el cerebro de la app, usare Langgraph
# - Con un flujo de nodo sencillo, pero que es flexible
#   para agregar o quitar nodos segun sea necesario
# ---------------------------------------------------------------------------


from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from app.config import LLM_MODEL
from app.rag.rerank import rerank_chunks
from app.rag.retrieval import retrieve_chunks

try:
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register
except Exception:
    register = None
    LangChainInstrumentor = None


# ---------------------------------------------------------------------------
# Observabilidad con phoenix
# ---------------------------------------------------------------------------
if register and LangChainInstrumentor:
    try:
        tracer_provider = register(
            project_name="QuoteFinder",
            endpoint="http://localhost:6006/v1/traces",
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    except Exception:
        # Si observabilidad no está disponible, el workflow sigue funcionando.
        tracer_provider = None
else:
    tracer_provider = None


# ---------------------------------------------------------------------------
# Estado global del grafo
# ---------------------------------------------------------------------------
# Defino el estado global
class State(TypedDict):
    query: str
    retrieved_chunks: list[str]
    reranked_chunks: list[str]
    final_answer: str


# Inicializo el grafo con el estado global
graph = StateGraph(State)


# ---------------------------------------------------------------------------
# Nodos del grafo
# ---------------------------------------------------------------------------


def retrieve_node(state: State) -> State:
    state["retrieved_chunks"] = retrieve_chunks(state["query"])
    return state


def rerank_node(state: State) -> State:
    state["reranked_chunks"] = rerank_chunks(state["query"], state["retrieved_chunks"])
    return state


def response_node(state: State) -> State:
    query = state["query"]
    chunks_reranked = state["reranked_chunks"]

    prompt = f"""
    <rol>Eres un asistente de investigación que ayuda a formatear citas textuales de entrevistas.</rol>
    <tarea>A partir de los siguientes chunks de texto, extrae las citas textuales y presentalas en formato Markdown, indicando la fuente de la cita</tarea>
    <chunks>{chunks_reranked}</chunks>
    <query_usuario>{query}</query_usuario>
    <reglas>Debes responder solo con las citas textuales encontradas en los chunks, sin agregar información adicional. Cada cita debe ir acompañada de su fuente, que se encuentra en la metadata de cada chunk bajo la clave "fuente". Si no hay citas que sirvan para responder al usuario, indica que no se encontraron citas.</reglas>
    <ejemplo_formato>
    > CITA TEXTUAL
    Fuente: fuente de la cita
    </ejemplo_formato>
    """

    # Instancio ollama
    llm = ChatOllama(model=LLM_MODEL, temperature=0)
    llm_response = llm.invoke(prompt)
    state["final_answer"] = llm_response.content
    return state


# ---------------------------------------------------------------------------
# Agregar nodos del grafo
# ---------------------------------------------------------------------------
graph.add_node("Retrieve", retrieve_node)
graph.add_node("Rerank", rerank_node)
graph.add_node("Response", response_node)


# ---------------------------------------------------------------------------
# Agregar edges al grafo
# ---------------------------------------------------------------------------
graph.add_edge(START, "Retrieve")
graph.add_edge("Retrieve", "Rerank")
graph.add_edge("Rerank", "Response")
graph.add_edge("Response", END)


# ---------------------------------------------------------------------------
# Compilar el grafo para su ejecución
# ---------------------------------------------------------------------------
graph_compiled = graph.compile()

# def main():
#     # Ejemplo de uso del grafo
#     initial_state: State = {
#         "query": "busca temas sobre andinismo",
#         "retrieved_chunks": [],
#         "reranked_chunks": [],
#         "final_answer": "",
#     }
#     compiled = graph.compile()
#     final_state = compiled.invoke(initial_state)
#     # print(final_state["retrieved_chunks"])
#     print("#" * 80)
#     print(final_state["reranked_chunks"])
#     print("#" * 80)
#     print("#" * 80)
#     print(final_state["final_answer"])


# if __name__ == "__main__":
#     main()
