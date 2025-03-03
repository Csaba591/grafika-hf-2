//=============================================================================================
// Computer Graphics Sample Program: GPU ray casting
//=============================================================================================
#include "framework.h"

// Most accurate float representation of pi (0x40490FDB)
const float M_PI_F = 3.1415927410125732421875E0F;

// vertex shader in GLSL
const char *vertexSource =
        R"(
    #version 450
    precision highp float;

    uniform vec3 wLookAt, wRight, wUp;          // pos of eye

    layout(location = 0) in vec2 cCamWindowVertex;	// Attrib Array 0
    out vec3 p;

    void main() {
        gl_Position = vec4(cCamWindowVertex, 0, 1);
        p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
    }
)";
// fragment shader in GLSL
const char *fragmentSource =
        R"(
    #version 450
    precision highp float;

    struct Material {
        vec3 ka, kd, ks;
        float  shininess;
        vec3 F0;
        int rough, reflective;
    };

    struct Light {
        vec3 direction;
        vec3 Le, La;
    };

    struct Sphere {
        vec3 center;
        float radius;
    };

    struct Hit {
        float t;
        vec3 position, normal;
        int mat;	// material index
    };

    struct Ray {
        vec3 start, dir;
    };

    const int nMaxObjects = 500;

    uniform vec3 wEye;
    uniform Light light;
    uniform Material materials[2];  // diffuse, specular, ambient ref
    uniform int nObjects;
    uniform Sphere objects[nMaxObjects];

    in  vec3 p;					// point on camera window corresponding to the pixel
    out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

    Hit intersect(const Sphere object, const Ray ray) {
        Hit hit;
        hit.t = -1;
        vec3 dist = ray.start - object.center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - object.radius * object.radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrt(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - object.center) / object.radius;
        return hit;
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        bestHit.t = -1;
        for (int o = 0; o < nObjects; o++) {
            Hit hit = intersect(objects[o], ray); //  hit.t < 0 if no intersection
            if (o < nObjects/2) hit.mat = 0;	 // half of the objects are rough
            else			    hit.mat = 1;     // half of the objects are reflective
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (int o = 0; o < nObjects; o++) if (intersect(objects[o], ray).t > 0) return true; //  hit.t < 0 if no intersection
        return false;
    }

    vec3 Fresnel(vec3 F0, float cosTheta) {
        return F0 + (vec3(1, 1, 1) - F0) * pow(cosTheta, 5);
    }

    const float epsilon = 0.0001f;
    const int maxdepth = 5;

    vec3 trace(Ray ray) {
        vec3 weight = vec3(1, 1, 1);
        vec3 outRadiance = vec3(0, 0, 0);
        for(int d = 0; d < maxdepth; d++) {
            Hit hit = firstIntersect(ray);
            if (hit.t < 0) return weight * light.La;
            if (materials[hit.mat].rough == 1) {
                outRadiance += weight * materials[hit.mat].ka * light.La;
                Ray shadowRay;
                shadowRay.start = hit.position + hit.normal * epsilon;
                shadowRay.dir = light.direction;
                float cosTheta = dot(hit.normal, light.direction);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
                    outRadiance += weight * light.Le * materials[hit.mat].kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + light.direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0) outRadiance += weight * light.Le * materials[hit.mat].ks * pow(cosDelta, materials[hit.mat].shininess);
                }
            }

            if (materials[hit.mat].reflective == 1) {
                weight *= Fresnel(materials[hit.mat].F0, dot(-ray.dir, hit.normal));
                ray.start = hit.position + hit.normal * epsilon;
                ray.dir = reflect(ray.dir, hit.normal);
            } else return outRadiance;
        }
    }

    void main() {
        Ray ray;
        ray.start = wEye;
        ray.dir = normalize(p - wEye);
        fragmentColor = vec4(trace(ray), 1);
    }
)";

//---------------------------
struct Material {
    //---------------------------
    vec3 ka, kd, ks;
    float shininess{};
    vec3 F0;
    int rough{}, reflective{};
};

//---------------------------
struct RoughMaterial : Material {
    //---------------------------
    RoughMaterial(const vec3 _kd, const vec3 _ks, const float _shininess) {
        ka = _kd * M_PI_F;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
        rough = 1;
        reflective = 0;
    }
};

//---------------------------
struct SmoothMaterial : Material {
    //---------------------------
    explicit SmoothMaterial(const vec3 _F0) {
        F0 = _F0;
        rough = 0;
        reflective = 1;
    }
};

//---------------------------
class Sphere {
    //---------------------------
    vec3 center;
    float radius;
public:
    const vec3 &getCenter() const {
        return center;
    }

    float getRadius() const {
        return radius;
    }

    Sphere(const vec3 &_center, const float _radius) {
        center = _center;
        radius = _radius;
    }
};

//---------------------------
class Camera {
    //---------------------------
    vec3 eye, lookat, right, up;
    float fov{};
public:
    const vec3 &getEye() const {
        return eye;
    }

    const vec3 &getLookat() const {
        return lookat;
    }

    const vec3 &getRight() const {
        return right;
    }

    const vec3 &getUp() const {
        return up;
    }

    void set(const vec3 _eye, const vec3 _lookat, const vec3 vup, const float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        const auto w = eye - lookat;
        const auto f = length(w);
        right = normalize(cross(vup, w)) * f * tanf(fov / 2);
        up = normalize(cross(w, right)) * f * tanf(fov / 2);
    }

    void Animate(const float dt) {
        eye = vec3((eye.x - lookat.x) * cosf(dt) + (eye.z - lookat.z) * sinf(dt) + lookat.x,
                   eye.y,
                   -(eye.x - lookat.x) * sinf(dt) + (eye.z - lookat.z) * cosf(dt) + lookat.z);
        set(eye, lookat, up, fov);
    }
};

//---------------------------
class Light {
    //---------------------------
    vec3 direction;
    vec3 Le, La;
public:
    const vec3 &getDirection() const {
        return direction;
    }

    const vec3 &getLe() const {
        return Le;
    }

    const vec3 &getLa() const {
        return La;
    }

    Light(const vec3 _direction, const vec3 _Le, const vec3 _La) {
        direction = normalize(_direction);
        Le = _Le;
        La = _La;
    }
};

//---------------------------
class Shader : public GPUProgram {
    //---------------------------
public:
    void setUniformMaterials(const std::vector<Material *> &materials) {
        for (unsigned int mat = 0; mat < materials.size(); mat++) {
            auto matIndex = std::to_string(mat);
            setUniform(materials[mat]->ka, "materials[" + matIndex + "].ka");
            setUniform(materials[mat]->kd, "materials[" + matIndex + "].kd");
            setUniform(materials[mat]->ks, "materials[" + matIndex + "].ks");
            setUniform(materials[mat]->shininess, "materials[" + matIndex + "].shininess");
            setUniform(materials[mat]->F0, "materials[" + matIndex + "].F0");
            setUniform(materials[mat]->rough, "materials[" + matIndex + "].rough");
            setUniform(materials[mat]->reflective, "materials[" + matIndex + "].reflective");
        }
    }

    void setUniformLight(Light *light) {
        setUniform(light->getLa(), "light.La");
        setUniform(light->getLe(), "light.Le");
        setUniform(light->getDirection(), "light.direction");
    }

    void setUniformCamera(const Camera &camera) {
        setUniform(camera.getEye(), "wEye");
        setUniform(camera.getLookat(), "wLookAt");
        setUniform(camera.getRight(), "wRight");
        setUniform(camera.getUp(), "wUp");
    }

    void setUniformObjects(const std::vector<Sphere *> &objects) {
        setUniform(static_cast<int>(objects.size()), "nObjects");
        for (unsigned int o = 0; o < objects.size(); o++) {
            auto objIndex = std::to_string(o);
            setUniform(objects[o]->getCenter(), "objects[" + objIndex + "].center");
            setUniform(objects[o]->getRadius(), "objects[" + objIndex + "].radius");
        }
    }
};

float rnd() {
    return static_cast<float>(rand()) / RAND_MAX;
}

//---------------------------
class Scene {
    //---------------------------
    std::vector<Sphere *> objects;
    std::vector<Light *> lights;
    Camera camera;
    std::vector<Material *> materials;
public:
    void build() {
        const auto eye = vec3(0, 0, 2);
        const auto vup = vec3(0, 1, 0);
        const auto lookat = vec3(0, 0, 0);
        const auto fov = 45 * M_PI_F / 180;
        camera.set(eye, lookat, vup, fov);

        lights.push_back(new Light(vec3(1, 1, 1), vec3(3, 3, 3), vec3(0.4F, 0.3F, 0.3F)));

        const vec3 kd(0.3F, 0.2F, 0.1F);
        const vec3 ks(10, 10, 10);
        materials.push_back(new RoughMaterial(kd, ks, 50));
        materials.push_back(new SmoothMaterial(vec3(0.9F, 0.85F, 0.8F)));

        for (auto i = 0; i < 500; i++) {
            objects.push_back(new Sphere(vec3(rnd() - 0.5F, rnd() - 0.5F, rnd() - 0.5F), rnd() * 0.1F));
        }
    }

    void setUniform(Shader *shader) {
        shader->setUniformObjects(objects);
        shader->setUniformMaterials(materials);
        shader->setUniformLight(lights[0]);
        shader->setUniformCamera(camera);
    }

    void Animate(const float dt) {
        camera.Animate(dt);
    }
};

Shader shader; // vertex and fragment shaders
Scene scene;

//---------------------------
class FullScreenTexturedQuad {
    //---------------------------
    unsigned int vao = 0; // vertex array object id and texture id
public:
    void create() {
        glGenVertexArrays(1, &vao); // create 1 vertex array object
        glBindVertexArray(vao); // make it active

        unsigned int vbo; // vertex buffer objects
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = {-1, -1, 1, -1, 1, 1, -1, 1}; // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof vertexCoords, static_cast<const void *>(vertexCoords),
                     GL_STATIC_DRAW); // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr); // stride and offset: it is tightly packed
    }

    void Draw() const {
        glBindVertexArray(vao); // make the vao and its vbos active playing the role of the data source
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4); // draw two triangles forming a quad
    }
};

FullScreenTexturedQuad fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();
    fullScreenTexturedQuad.create();

    // create program for the GPU
    shader.create(vertexSource, fragmentSource, "fragmentColor");
    shader.Use();
}

// Window has become invalid: Redraw
void onDisplay() {
    static auto nFrames = 0;
    nFrames++;
    static auto tStart = glutGet(GLUT_ELAPSED_TIME);
    const auto tEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("%d msec\r", (tEnd - tStart) / nFrames);

    glClearColor(1.0F, 0.5F, 0.8F, 1.0F); // background color
    glClear(static_cast<GLbitfield>(GL_COLOR_BUFFER_BIT) |
            static_cast<GLbitfield>(GL_DEPTH_BUFFER_BIT)); // clear the screen

    scene.setUniform(&shader);
    fullScreenTexturedQuad.Draw();

    glutSwapBuffers(); // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char /*key*/, int /*pX*/, int /*pY*/) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char /*key*/, int /*pX*/, int /*pY*/) {
}

// Mouse click event
void onMouse(int /*button*/, int /*state*/, int /*pX*/, int /*pY*/) {
}

// Move mouse with key pressed
void onMouseMotion(int /*pX*/, int /*pY*/) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    scene.Animate(0.01F);
    glutPostRedisplay();
}
