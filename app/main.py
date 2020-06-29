from ariadne import ObjectType, QueryType, MutationType, ScalarType, gql, make_executable_schema, load_schema_from_path
from ariadne.asgi import GraphQL
from asgi_lifespan import Lifespan, LifespanMiddleware

from dotenv import load_dotenv


from app.resolvers import resolvers


# Load environment variables
load_dotenv()




# Define types using Schema Definition Language (https://graphql.org/learn/schema/)
# Wrapping string in gql function provides validation and better error traceback

type_defs = load_schema_from_path("app/schema")

all_resolvers = []

# Map resolver functions to Query fields using QueryType
if 'Query' in resolvers and len(resolvers['Query']) > 0:
    query = QueryType()
    for q in resolvers['Query']:
        query.set_field(q, resolvers['Query'][q])
    all_resolvers.append(query)

# Map resolver functions to custom type fields using ObjectType
if 'Object' in resolvers and len(resolvers['Object']) > 0:
    for obj in resolvers['Object']:
        objType = ObjectType(obj)
        for fld in resolvers['Object'][obj]:
            objType.set_field(fld, resolvers['Object'][obj][fld])
        all_resolvers.append(objType)

# Map resolver functions to Mutation fields using MutationType
if 'Mutation' in resolvers and len(resolvers['Mutation']) > 0:
    mutation = MutationType()
    for m in resolvers['Mutation']:
        mutation.set_field(m, resolvers['Mutation'][m])
    all_resolvers.append(mutation)

# Map resolver functions to custom scalars using ScalarType
if 'Scalar' in resolvers and len(resolvers['Scalar']) > 0:
    for scalar in resolvers['Scalar']:
        scalarType = ScalarType(scalar)
        scalarType.set_serializer(resolvers['Scalar'][scalar]['serializer'])
        scalarType.set_literal_parser(resolvers['Scalar'][scalar]['literal_parser'])
        scalarType.set_value_parser(resolvers['Scalar'][scalar]['value_parser'])
        all_resolvers.append(scalarType)

# Create executable GraphQL schema
schema = make_executable_schema(type_defs, all_resolvers)


# --- ASGI app

# Create an ASGI app using the schema, running in debug mode
# Set context with authenticated graphql client.
app = GraphQL(
    schema, debug=True, context_value={})

# 'Lifespan' is a standalone ASGI app.
# It implements the lifespan protocol,
# and allows registering lifespan event handlers.
lifespan = Lifespan()


@lifespan.on_event("startup")
async def startup():
    print("Starting up...")
    print("... done!")


@lifespan.on_event("shutdown")
async def shutdown():
    print("Shutting down...")
    print("... done!")

# 'LifespanMiddleware' returns an ASGI app.
# It forwards lifespan requests to 'lifespan',
# and anything else goes to 'app'.
app = LifespanMiddleware(app, lifespan=lifespan)
