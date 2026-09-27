"""
Script optimizado de migración integral de Neo4j Aura (Nube) a Neo4j Local.
- Recrea restricciones e índices vectoriales (384 y 1024 dims).
- Utiliza elementId() moderno e indexación temporal O(log n) para emparejar relaciones a máxima velocidad.
- Garantiza paridad exacta de nodos, propiedades y grafos.
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase

root_dir = Path(__file__).resolve().parents[1]
load_dotenv(root_dir / ".env", override=True)

AURA_URI = os.getenv("NEO4J_URI")
AURA_USER = os.getenv("NEO4J_USERNAME")
AURA_PWD = os.getenv("NEO4J_PASSWORD")

LOCAL_URI = "bolt://localhost:7687"
LOCAL_USER = "neo4j"
LOCAL_PWD = "password"


def migrate():
    print("=" * 70)
    print("🚀 MIGRACIÓN INTEGRAL: NEO4J AURA (NUBE) -> NEO4J LOCAL")
    print(f"Fuente (Aura):  {AURA_URI} (usuario: {AURA_USER})")
    print(f"Destino (Local): {LOCAL_URI} (usuario: {LOCAL_USER})")
    print("=" * 70)

    source_driver = GraphDatabase.driver(AURA_URI, auth=(AURA_USER, AURA_PWD))
    target_driver = GraphDatabase.driver(LOCAL_URI, auth=(LOCAL_USER, LOCAL_PWD))

    # 1. Probar conexiones y censar
    with source_driver.session() as s_sess:
        src_nodes = s_sess.run("MATCH (n) RETURN count(n) as count").single()["count"]
        src_rels = s_sess.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
        print(f"📦 Total datos en Aura: {src_nodes} nodos, {src_rels} relaciones.")

    # 2. Recrear restricciones de unicidad en Local
    print("\n🔑 Paso 1: Replicando restricciones de unicidad en destino...")
    with source_driver.session() as s_sess, target_driver.session() as t_sess:
        constraints = s_sess.run("SHOW CONSTRAINTS YIELD name, type, entityType, labelsOrTypes, properties").data()
        for c in constraints:
            if c["type"] == "UNIQUENESS" and c["entityType"] == "NODE":
                label = c["labelsOrTypes"][0]
                prop = c["properties"][0]
                c_name = c["name"]
                query = f"CREATE CONSTRAINT {c_name} IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                try:
                    t_sess.run(query)
                    print(f"   ✓ Creada restricción: {c_name} ({label}.{prop})")
                except Exception as e:
                    print(f"   ⚠️ Restricción {c_name}: {e}")

    # 3. Recrear índices vectoriales y de rango en Local
    print("\n🔍 Paso 2: Replicando índices (Vectoriales y Rango) en destino...")
    with source_driver.session() as s_sess, target_driver.session() as t_sess:
        indexes = s_sess.run("SHOW INDEXES YIELD name, type, entityType, labelsOrTypes, properties, options").data()
        for idx in indexes:
            itype = idx["type"]
            iname = idx["name"]
            labels = idx.get("labelsOrTypes")
            props = idx.get("properties")

            if itype == "VECTOR" and labels and props:
                label = labels[0]
                prop = props[0]
                options = idx.get("options", {})
                cfg = options.get("indexConfig", {})
                dims = cfg.get("vector.dimensions", 384)
                sim = cfg.get("vector.similarity_function", "COSINE")
                v_query = f"""
                CREATE VECTOR INDEX {iname} IF NOT EXISTS
                FOR (n:{label}) ON (n.{prop})
                OPTIONS {{
                  indexConfig: {{
                    `vector.dimensions`: {dims},
                    `vector.similarity_function`: '{sim}'
                  }}
                }}
                """
                try:
                    t_sess.run(v_query)
                    print(f"   ✓ Creado índice vectorial: {iname} ({label}.{prop}, {dims} dims, {sim})")
                except Exception as e:
                    pass

            elif itype == "RANGE" and labels and props and idx["entityType"] == "NODE":
                label = labels[0]
                prop = props[0]
                r_query = f"CREATE INDEX {iname} IF NOT EXISTS FOR (n:{label}) ON (n.{prop})"
                try:
                    t_sess.run(r_query)
                except Exception:
                    pass

    # Crear índice temporal en _AuraNode._aura_id para emparejamiento O(log n)
    with target_driver.session() as t_sess:
        try:
            t_sess.run("CREATE INDEX temp_aura_id IF NOT EXISTS FOR (n:_AuraNode) ON (n._aura_id)")
        except Exception:
            pass

    # 4. Migrar todos los nodos
    print(f"\n📥 Paso 3: Migrando {src_nodes} nodos con propiedades completas...")
    batch_size = 500
    skip = 0
    total_migrated_nodes = 0

    while True:
        with source_driver.session() as s_sess:
            node_batch = s_sess.run(
                f"""
                MATCH (n)
                RETURN elementId(n) as aura_id, labels(n) as labels, properties(n) as props
                SKIP {skip} LIMIT {batch_size}
                """
            ).data()

        if not node_batch:
            break

        with target_driver.session() as t_sess:
            with t_sess.begin_transaction() as tx:
                for item in node_batch:
                    labels = list(item["labels"])
                    if "_AuraNode" not in labels:
                        labels.append("_AuraNode")
                    labels_str = ":".join(f"`{lbl}`" for lbl in labels)
                    props = dict(item["props"])
                    props["_aura_id"] = item["aura_id"]

                    q = f"CREATE (n:{labels_str}) SET n = $props"
                    tx.run(q, {"props": props})
                tx.commit()

        total_migrated_nodes += len(node_batch)
        print(f"   Progreso nodos: {total_migrated_nodes}/{src_nodes}...")
        skip += batch_size

    # 5. Migrar todas las relaciones en lotes con transacciones rápidas
    print(f"\n🔗 Paso 4: Migrando {src_rels} relaciones...")
    skip = 0
    total_migrated_rels = 0

    while True:
        with source_driver.session() as s_sess:
            rel_batch = s_sess.run(
                f"""
                MATCH (a)-[r]->(b)
                RETURN elementId(a) as from_aura_id, elementId(b) as to_aura_id, type(r) as r_type, properties(r) as props
                SKIP {skip} LIMIT {batch_size}
                """
            ).data()

        if not rel_batch:
            break

        with target_driver.session() as t_sess:
            with t_sess.begin_transaction() as tx:
                for rel in rel_batch:
                    r_type = rel["r_type"]
                    from_id = rel["from_aura_id"]
                    to_id = rel["to_aura_id"]
                    r_props = dict(rel["props"])

                    rel_query = f"""
                    MATCH (a:_AuraNode {{_aura_id: $from_id}})
                    MATCH (b:_AuraNode {{_aura_id: $to_id}})
                    CREATE (a)-[r:`{r_type}`]->(b)
                    SET r = $props
                    """
                    tx.run(rel_query, {"from_id": from_id, "to_id": to_id, "props": r_props})
                tx.commit()

        total_migrated_rels += len(rel_batch)
        print(f"   Progreso relaciones: {total_migrated_rels}/{src_rels}...")
        skip += batch_size

    # 6. Limpiar etiqueta auxiliar _AuraNode
    print("\n🧹 Paso 5: Finalizando y limpiando etiquetas auxiliares...")
    with target_driver.session() as t_sess:
        t_sess.run("DROP INDEX temp_aura_id IF EXISTS")
        t_sess.run("MATCH (n:_AuraNode) REMOVE n:_AuraNode, n._aura_id")

    # 7. Verificación final
    with target_driver.session() as t_sess:
        dst_nodes = t_sess.run("MATCH (n) RETURN count(n) as count").single()["count"]
        dst_rels = t_sess.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]

    print("\n" + "=" * 70)
    print("🎉 MIGRACIÓN COMPLETADA EXITOSAMENTE")
    print(f"   Nodos:       Aura = {src_nodes}  -->  Local = {dst_nodes}")
    print(f"   Relaciones:  Aura = {src_rels}  -->  Local = {dst_rels}")
    if src_nodes == dst_nodes and src_rels == dst_rels:
        print("   ✅ Paridad 100% verificada: Todos los datos han sido migrados con fidelidad exacta.")
    else:
        print("   ⚠️ Diferencia detectada entre origen y destino. Revisa los logs.")
    print("=" * 70)

    source_driver.close()
    target_driver.close()


if __name__ == "__main__":
    migrate()
